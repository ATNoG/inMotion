//! The `LeJEPA` classifier, reimplemented for inference.
//!
//! Mirrors `LeJEPAEncoder.forward` and `LeJEPAClassifier.forward` exactly:
//!
//! ```text
//! z = tokenizer(x) + pos_embed
//! for each layer:                       # norm_first = true
//!     z = z + self_attn(norm1(z))
//!     z = z + linear2(gelu(linear1(norm2(z))))
//! z = gelu(proj(z))
//! z = batch_norm(z)                     # over the d_model axis
//! w = softmax(attn(z), dim=T)
//! pooled = sum(z * w, dim=T)
//! logits = head(pooled)                 # linear -> gelu -> linear
//! ```
//!
//! Batch size is fixed at 1. The device classifies one 10-second window at a
//! time, so carrying a batch axis would cost memory for no benefit, and it makes
//! the attention layout unambiguous.

use crate::artifact::Artifact;
use crate::ops;
use crate::tensor::Tensor;

/// Architecture hyperparameters, read from the artifact.
#[derive(Debug, Clone)]
pub struct LeJepaConfig {
    /// Timesteps per window.
    pub seq_len: usize,
    /// Input channels per timestep.
    pub in_channels: usize,
    /// Transformer width; also the width of the final pooling head.
    pub d_model: usize,
    /// Attention heads; cannot be inferred from parameter shapes, so it is read
    /// from the artifact and validated against `d_model`.
    pub n_heads: usize,
    /// Number of `TransformerEncoderLayer` blocks.
    pub num_layers: usize,
    /// Output classes.
    pub num_classes: usize,
}

impl LeJepaConfig {
    /// Derive the shape parameters from the artifact's tensor index.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, String> {
        let d_model = *artifact
            .shape_of("encoder.tokenizer.weight")
            .and_then(|s| s.first())
            .ok_or("missing encoder.tokenizer.weight")?;
        let in_channels = artifact.meta.in_channels;
        // head.3 is the final classification projection: (num_classes, hidden).
        let num_classes = *artifact
            .shape_of("head.3.weight")
            .and_then(|s| s.first())
            .ok_or("missing head.3.weight")?;
        // Read the layer count rather than probing until a lookup fails. Probing
        // silently accepts a gap in the numbering (layers 0, 1, 3 would count as
        // two) and allocates a formatted key per attempt.
        let num_layers = artifact
            .meta
            .arch
            .get("num_layers")
            .and_then(serde_json::Value::as_u64)
            .ok_or("artifact does not record `num_layers`")? as usize;
        if num_layers == 0 {
            return Err("artifact records num_layers = 0".into());
        }
        // Every recorded layer must actually be present.
        for l in 0..num_layers {
            let key = format!("encoder.transformer.layers.{l}.norm1.weight");
            if artifact.shape_of(&key).is_none() {
                return Err(format!("artifact is missing {key}"));
            }
        }
        Ok(Self {
            seq_len: artifact.meta.seq_len,
            in_channels,
            d_model,
            n_heads: infer_heads(artifact, d_model)?,
            num_layers,
            num_classes,
        })
    }
}

/// Read the attention head count from the artifact.
///
/// This cannot be inferred from the weights: `nhead` changes no parameter shape,
/// only how the projected dimensions are grouped. Guessing it would produce a
/// model that runs and returns confident wrong answers, so an absent value is an
/// error rather than a default.
fn infer_heads(artifact: &Artifact, d_model: usize) -> Result<usize, String> {
    let heads = artifact
        .meta
        .arch
        .get("nhead")
        .and_then(serde_json::Value::as_u64)
        .ok_or("artifact does not record `nhead`; re-export it from the registry entry")?
        as usize;
    if heads == 0 || !d_model.is_multiple_of(heads) {
        return Err(format!("nhead {heads} does not divide d_model {d_model}"));
    }
    Ok(heads)
}

/// A loaded `LeJEPA` classifier, holding dequantized weights.
pub struct LeJepa {
    /// Shape parameters, validated against the loaded weights.
    pub config: LeJepaConfig,
    tokenizer_w: Tensor,
    tokenizer_b: Tensor,
    pos_embed: Tensor,
    layers: Vec<Layer>,
    proj_w: Tensor,
    proj_b: Tensor,
    bn_w: Tensor,
    bn_b: Tensor,
    bn_mean: Tensor,
    bn_var: Tensor,
    attn_w: Tensor,
    attn_b: Tensor,
    head0_w: Tensor,
    head0_b: Tensor,
    head3_w: Tensor,
    head3_b: Tensor,
    eps_ln: f32,
    eps_bn: f32,
}

struct Layer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    ff1_w: Tensor,
    ff1_b: Tensor,
    ff2_w: Tensor,
    ff2_b: Tensor,
}

impl LeJepa {
    /// Load every weight from the artifact, dequantizing as needed.
    pub fn load(artifact: &Artifact) -> Result<Self, String> {
        let config = LeJepaConfig::from_artifact(artifact)?;
        let t = |name: &str| artifact.tensor(name);

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let p = format!("encoder.transformer.layers.{l}");
            layers.push(Layer {
                norm1_w: t(&format!("{p}.norm1.weight"))?,
                norm1_b: t(&format!("{p}.norm1.bias"))?,
                q_w: t(&format!("{p}.self_attn.q.weight"))?,
                q_b: t(&format!("{p}.self_attn.q.bias"))?,
                k_w: t(&format!("{p}.self_attn.k.weight"))?,
                k_b: t(&format!("{p}.self_attn.k.bias"))?,
                v_w: t(&format!("{p}.self_attn.v.weight"))?,
                v_b: t(&format!("{p}.self_attn.v.bias"))?,
                out_w: t(&format!("{p}.self_attn.out_proj.weight"))?,
                out_b: t(&format!("{p}.self_attn.out_proj.bias"))?,
                norm2_w: t(&format!("{p}.norm2.weight"))?,
                norm2_b: t(&format!("{p}.norm2.bias"))?,
                ff1_w: t(&format!("{p}.linear1.weight"))?,
                ff1_b: t(&format!("{p}.linear1.bias"))?,
                ff2_w: t(&format!("{p}.linear2.weight"))?,
                ff2_b: t(&format!("{p}.linear2.bias"))?,
            });
        }

        Ok(Self {
            eps_ln: artifact.meta.eps.layernorm,
            eps_bn: artifact.meta.eps.batchnorm,
            config,
            tokenizer_w: t("encoder.tokenizer.weight")?,
            tokenizer_b: t("encoder.tokenizer.bias")?,
            pos_embed: t("encoder.pos_embed")?,
            layers,
            proj_w: t("encoder.proj.0.weight")?,
            proj_b: t("encoder.proj.0.bias")?,
            bn_w: t("encoder.bn.weight")?,
            bn_b: t("encoder.bn.bias")?,
            bn_mean: t("encoder.bn.running_mean")?,
            bn_var: t("encoder.bn.running_var")?,
            attn_w: t("attn.weight")?,
            attn_b: t("attn.bias")?,
            head0_w: t("head.0.weight")?,
            head0_b: t("head.0.bias")?,
            head3_w: t("head.3.weight")?,
            head3_b: t("head.3.bias")?,
        })
    }

    /// Run one window, `x: (seq_len, in_channels)`; returns `(num_classes,)` logits.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (t, c) = x.as_2d();
        assert_eq!(
            c, self.config.in_channels,
            "expected {} input channels",
            self.config.in_channels
        );
        assert_eq!(
            t, self.config.seq_len,
            "expected {} timesteps",
            self.config.seq_len
        );

        // tokenizer + positional embedding
        let mut z = ops::linear(x, &self.tokenizer_w, &self.tokenizer_b);
        z = ops::add(&z, &self.pos_embed);

        for layer in &self.layers {
            let normed = ops::layer_norm(&z, &layer.norm1_w, &layer.norm1_b, self.eps_ln);
            let attn = ops::self_attention(
                &normed,
                &layer.q_w,
                &layer.q_b,
                &layer.k_w,
                &layer.k_b,
                &layer.v_w,
                &layer.v_b,
                &layer.out_w,
                &layer.out_b,
                self.config.n_heads,
            );
            z = ops::add(&z, &attn);

            let normed = ops::layer_norm(&z, &layer.norm2_w, &layer.norm2_b, self.eps_ln);
            let ff = ops::linear(&normed, &layer.ff1_w, &layer.ff1_b);
            let ff = ops::gelu(&ff);
            let ff = ops::linear(&ff, &layer.ff2_w, &layer.ff2_b);
            z = ops::add(&z, &ff);
        }

        // projection head, then BatchNorm over the d_model axis
        let proj = ops::linear(&z, &self.proj_w, &self.proj_b);
        let proj = ops::gelu(&proj);
        let transposed = ops::transpose_2d(&proj);
        let normed = ops::batch_norm_eval(
            &transposed,
            &self.bn_w,
            &self.bn_b,
            &self.bn_mean,
            &self.bn_var,
            self.eps_bn,
        );
        // Transpose back; reinterpreting (C, T) memory as (T, C) would reorder
        // every element.
        let z = ops::transpose_2d(&normed);

        // attention pooling over timesteps
        let scores = ops::linear(&z, &self.attn_w, &self.attn_b);
        let weights = ops::softmax_first_axis(&scores);
        let mut pooled = vec![0.0f32; self.config.d_model];
        for i in 0..t {
            let weight = weights.data[i];
            let row = &z.data[i * self.config.d_model..(i + 1) * self.config.d_model];
            for (acc, value) in pooled.iter_mut().zip(row) {
                *acc += value * weight;
            }
        }
        let pooled = Tensor::new(pooled, vec![1, self.config.d_model]);

        // classification head
        let hidden = ops::linear(&pooled, &self.head0_w, &self.head0_b);
        let hidden = ops::gelu(&hidden);
        ops::linear(&hidden, &self.head3_w, &self.head3_b)
    }
}

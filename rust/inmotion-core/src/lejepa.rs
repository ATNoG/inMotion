//! The LeJEPA classifier, reimplemented for inference.
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
    pub seq_len: usize,
    pub in_channels: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub num_layers: usize,
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
        let num_layers = (0..)
            .take_while(|&l| artifact.shape_of(&format!("encoder.transformer.layers.{l}.norm1.weight")).is_some())
            .count();
        if num_layers == 0 {
            return Err("no transformer layers found in the artifact".into());
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

/// Recover the head count from the attention layout.
///
/// PyTorch does not store `nhead` in the weights, and the head count does not
/// change any parameter shape, so it cannot be derived from shapes alone. The
/// artifact therefore records the architecture explicitly; this function reads it
/// and falls back only when the value is absent.
fn infer_heads(artifact: &Artifact, d_model: usize) -> Result<usize, String> {
    if let Some(value) = artifact.meta.arch.get("nhead").and_then(|v| v.as_u64()) {
        return Ok(value as usize);
    }
    // A sensible default that divides d_model; recorded in meta for real runs.
    for candidate in [8usize, 4, 2, 1] {
        if d_model % candidate == 0 {
            return Ok(candidate);
        }
    }
    Ok(1)
}

/// A loaded LeJEPA classifier, holding dequantized weights.
pub struct LeJepa {
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
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (t, c) = x.as_2d();
        assert_eq!(c, self.config.in_channels, "expected {} input channels", self.config.in_channels);
        assert_eq!(t, self.config.seq_len, "expected {} timesteps", self.config.seq_len);

        // tokenizer + positional embedding
        let mut z = ops::linear(x, &self.tokenizer_w, &self.tokenizer_b);
        z = ops::add(&z, &self.pos_embed);

        for layer in &self.layers {
            let normed = ops::layer_norm(&z, &layer.norm1_w, &layer.norm1_b, self.eps_ln);
            let attn = ops::self_attention(
                &normed,
                &layer.q_w, &layer.q_b,
                &layer.k_w, &layer.k_b,
                &layer.v_w, &layer.v_b,
                &layer.out_w, &layer.out_b,
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
            for d in 0..self.config.d_model {
                pooled[d] += z.data[i * self.config.d_model + d] * weights.data[i];
            }
        }
        let pooled = Tensor::new(pooled, vec![1, self.config.d_model]);

        // classification head
        let hidden = ops::linear(&pooled, &self.head0_w, &self.head0_b);
        let hidden = ops::gelu(&hidden);
        ops::linear(&hidden, &self.head3_w, &self.head3_b)
    }
}

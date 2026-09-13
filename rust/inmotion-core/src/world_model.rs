//! The transformer world models: `LeJEPA`, `T-JEPA`, `TS-JEPA` and `CF-JEPA`.
//!
//! All four are a pre-norm transformer stack, a single-score attention pool over
//! tokens, and a two-layer classification head. They differ only in how a window
//! becomes tokens and in the two small stages around the stack:
//!
//! * **the tokenizer.**
//!   * `LeJEPA` projects each timestep with a `Linear` (`encoder.tokenizer`).
//!   * `TS-JEPA` and `CF-JEPA` convolve non-overlapping patches with a
//!     `Conv1d(kernel = stride = patch_size)` (`encoder.tokenizer.proj`).
//!   * `T-JEPA` scales each input slot elementwise by its own weight row
//!     (`encoder.tokenizer.weight`), adds a per-slot bias and a feature-index
//!     embedding, and prepends `n_reg` `[REG]` slots whose input value is the
//!     constant 1.0.
//! * **the projection between the tokenizer and the transformer**, which only
//!   `T-JEPA` has (`encoder.input_proj`).
//! * **the tail between the transformer and the pool.** `T-JEPA`, `TS-JEPA` and
//!   `CF-JEPA` apply a final `LayerNorm` (`encoder.norm`); `LeJEPA` applies
//!   `Linear + GELU` (`encoder.proj`) then `BatchNorm1d` over the width axis
//!   (`encoder.bn`) — the projection the `LeWM` paper requires for `SIGReg`.
//!
//! Every one of those choices is recorded in the artifact, so one implementation
//! serves all four families and the runtime never guesses which it holds:
//!
//! ```text
//! z = tokenizer(x) [+ pos_embed] [+ input_proj]
//! for each layer:                       # norm_first = true
//!     z = z + self_attn(norm1(z))
//!     z = z + linear2(gelu(linear1(norm2(z))))
//! z = tail(z)                           # LayerNorm, or proj + gelu + batchnorm
//! z = z[n_reg:]                         # T-JEPA's [REG] slots are not pooled
//! w = softmax(attn(z), dim = tokens)
//! pooled = sum(z * w, dim = tokens)
//! logits = head(pooled)                 # linear -> gelu -> linear
//! ```
//!
//! Batch size is fixed at 1: the device classifies one 10-second window at a
//! time, so a batch axis would cost memory and make the attention layout
//! ambiguous for no benefit.

use crate::artifact::Artifact;
use crate::ops::{self, PackedWeight};
use crate::tensor::Tensor;

/// How a `(seq_len, in_channels)` window becomes `(n_tokens, d_model)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tokenizer {
    /// One token per timestep: `encoder.tokenizer` is a `Linear(in, d_model)`.
    Linear,
    /// Non-overlapping patches: `encoder.tokenizer.proj` is a
    /// `Conv1d(in, d_model, kernel, stride)`.
    Patch {
        /// Convolution kernel width in timesteps.
        kernel: usize,
        /// Convolution stride in timesteps.
        stride: usize,
    },
    /// `T-JEPA`: `n_reg` `[REG]` slots followed by the flattened window, each
    /// slot scaled elementwise by its own row of `encoder.tokenizer.weight`.
    Feature {
        /// Timesteps times channels — the number of real input slots.
        total_features: usize,
    },
}

/// The stage between the transformer and the pooling head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderTail {
    /// A final `LayerNorm` over the width axis (`T-JEPA`, `TS-JEPA`, `CF-JEPA`).
    LayerNorm,
    /// `Linear + GELU` then `BatchNorm1d` over the width axis (`LeJEPA`).
    ProjectBatchNorm,
}

/// Architecture hyperparameters, read from the artifact.
#[derive(Debug, Clone)]
pub struct Config {
    /// Timesteps per window.
    pub seq_len: usize,
    /// Input channels per timestep.
    pub in_channels: usize,
    /// Transformer width.
    pub d_model: usize,
    /// Tokens the tokenizer produces, `n_reg` included. Recorded explicitly by
    /// the exporter and cross-checked against every tokenizer shape.
    pub n_tokens: usize,
    /// Leading `[REG]` slots, dropped before pooling. Zero for every family
    /// except `T-JEPA`.
    pub n_reg: usize,
    /// Attention heads; cannot be inferred from parameter shapes, so it is read
    /// from the artifact and validated against `d_model`.
    pub n_heads: usize,
    /// Number of `TransformerEncoderLayer` blocks.
    pub num_layers: usize,
    /// Output classes.
    pub num_classes: usize,
    /// The tokenizer branch this artifact expects.
    pub tokenizer: Tokenizer,
    /// The tail branch this artifact expects.
    pub tail: EncoderTail,
    /// Whether a `Linear(d_model, d_model)` projection sits between the
    /// tokenizer and the transformer (`T-JEPA` only).
    pub input_proj: bool,
}

/// Recover the conv stride from the patch count the artifact declares.
///
/// The tokenizer's stride is not a parameter shape, and assuming it equals the
/// kernel would silently accept an overlapping-patch model. The patch count is
/// recorded, so `(seq_len - kernel) / stride + 1` pins the stride down.
fn patch_stride(seq_len: usize, kernel: usize, n_tokens: usize) -> Result<usize, String> {
    let span = seq_len
        .checked_sub(kernel)
        .ok_or_else(|| format!("kernel {kernel} is longer than seq_len {seq_len}"))?;
    if n_tokens < 2 {
        return Err(format!(
            "cannot recover the patch stride from {n_tokens} token(s) \
             (seq_len {seq_len}, kernel {kernel})"
        ));
    }
    let steps = n_tokens - 1;
    let stride = span / steps;
    if stride == 0 || span % steps != 0 || span / stride + 1 != n_tokens {
        return Err(format!(
            "no integer stride maps seq_len {seq_len} with kernel {kernel} onto {n_tokens} patches"
        ));
    }
    Ok(stride)
}

/// Resolve the tokenizer branch from the recorded name, cross-checked against
/// the weights it names so a lying artifact fails at load rather than at
/// inference.
fn tokenizer_from_artifact(
    artifact: &Artifact,
    seq_len: usize,
    in_channels: usize,
    d_model: usize,
    n_tokens: usize,
    n_reg: usize,
) -> Result<Tokenizer, String> {
    match artifact.meta.str_field("tokenizer")? {
        "linear" => {
            if n_tokens != seq_len || n_reg != 0 {
                return Err(format!(
                    "linear tokenizer means one token per timestep and no [REG] slots, but the \
                     artifact declares n_tokens {n_tokens}, n_reg {n_reg} for seq_len {seq_len}"
                ));
            }
            let shape = artifact
                .shape_of("encoder.tokenizer.weight")
                .ok_or("tokenizer is `linear` but encoder.tokenizer.weight is missing")?;
            if shape.len() != 2 || shape[0] != d_model || shape[1] != in_channels {
                return Err(format!(
                    "encoder.tokenizer.weight {shape:?} does not project {in_channels} \
                     channels into {d_model}"
                ));
            }
            Ok(Tokenizer::Linear)
        }
        "patch" => {
            if n_reg != 0 {
                return Err(format!(
                    "patch tokenizer has no [REG] slots, but n_reg is {n_reg}"
                ));
            }
            let kernel = artifact.meta.usize_field("patch_size")?;
            let shape = artifact
                .shape_of("encoder.tokenizer.proj.weight")
                .ok_or("tokenizer is `patch` but encoder.tokenizer.proj.weight is missing")?;
            if shape.len() != 3 || shape[0] != d_model || shape[1] != in_channels {
                return Err(format!(
                    "encoder.tokenizer.proj.weight {shape:?} is not a {d_model} x \
                     {in_channels} x kernel convolution"
                ));
            }
            if shape[2] != kernel {
                return Err(format!(
                    "artifact says patch_size {kernel} but the conv kernel is {}",
                    shape[2]
                ));
            }
            Ok(Tokenizer::Patch {
                kernel,
                stride: patch_stride(seq_len, kernel, n_tokens)?,
            })
        }
        "feature" => {
            let total_features = artifact.meta.usize_field("total_features")?;
            if total_features != seq_len * in_channels {
                return Err(format!(
                    "artifact says total_features {total_features}, but seq_len {seq_len} x \
                     in_channels {in_channels} is {}",
                    seq_len * in_channels
                ));
            }
            if n_reg + total_features != n_tokens {
                return Err(format!(
                    "n_reg {n_reg} + total_features {total_features} != n_tokens {n_tokens}"
                ));
            }
            let weight = artifact
                .shape_of("encoder.tokenizer.weight")
                .ok_or("tokenizer is `feature` but encoder.tokenizer.weight is missing")?;
            if weight.len() != 2 || weight[0] != n_tokens || weight[1] != d_model {
                return Err(format!(
                    "encoder.tokenizer.weight {weight:?} is not one row per slot \
                     ({n_tokens} x {d_model})"
                ));
            }
            for name in [
                "encoder.tokenizer.bias",
                "encoder.tokenizer.feat_idx_embed.weight",
            ] {
                let shape = artifact
                    .shape_of(name)
                    .ok_or_else(|| format!("tokenizer is `feature` but {name} is missing"))?;
                if shape.as_slice() != [total_features, d_model] {
                    return Err(format!(
                        "{name} {shape:?} does not cover the {total_features} features at \
                         width {d_model}"
                    ));
                }
            }
            Ok(Tokenizer::Feature { total_features })
        }
        other => Err(format!("unknown tokenizer {other:?}")),
    }
}

/// Check that `encoder.pos_embed` is present exactly when the tokenizer adds one.
///
/// `T-JEPA` has none — its feature-index embedding is part of the tokenizer.
fn validate_pos_embed(
    artifact: &Artifact,
    tokenizer: Tokenizer,
    n_tokens: usize,
    d_model: usize,
) -> Result<(), String> {
    match (tokenizer, artifact.shape_of("encoder.pos_embed")) {
        (Tokenizer::Feature { .. }, Some(shape)) => Err(format!(
            "a feature tokenizer carries its own index embedding, but the artifact also holds \
             encoder.pos_embed {shape:?}"
        )),
        (Tokenizer::Feature { .. }, None) => Ok(()),
        (_, None) => Err(format!(
            "{tokenizer:?} tokenizer needs encoder.pos_embed, which is missing"
        )),
        (_, Some(shape)) => {
            if shape.len() != 3 || shape[0] != 1 || shape[1] != n_tokens || shape[2] != d_model {
                return Err(format!(
                    "encoder.pos_embed {shape:?} does not match (1, {n_tokens}, {d_model})"
                ));
            }
            Ok(())
        }
    }
}

/// Resolve the post-transformer stage from the recorded name, cross-checked
/// against the tensors it names.
fn tail_from_artifact(artifact: &Artifact, d_model: usize) -> Result<EncoderTail, String> {
    match artifact.meta.str_field("tail")? {
        "layernorm" => {
            if artifact.shape_of("encoder.norm.weight").is_none() {
                return Err("tail is `layernorm` but encoder.norm.weight is missing".into());
            }
            Ok(EncoderTail::LayerNorm)
        }
        "proj_bn" => {
            let proj = artifact
                .shape_of("encoder.proj.0.weight")
                .ok_or("tail is `proj_bn` but encoder.proj.0.weight is missing")?;
            if proj.len() != 2 || proj[0] != d_model || proj[1] != d_model {
                return Err(format!(
                    "encoder.proj.0.weight {proj:?} is not a {d_model} x {d_model} projection"
                ));
            }
            if artifact.shape_of("encoder.bn.weight").is_none() {
                return Err("tail is `proj_bn` but encoder.bn.weight is missing".into());
            }
            Ok(EncoderTail::ProjectBatchNorm)
        }
        other => Err(format!("unknown tail {other:?}")),
    }
}

impl Config {
    /// Derive the shape parameters from the artifact's tensor index.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, String> {
        let seq_len = artifact.meta.seq_len;
        let in_channels = artifact.meta.in_channels;
        let (d_model, n_tokens, n_reg) = (
            artifact.meta.usize_field("d_model")?,
            artifact.meta.usize_field("n_tokens")?,
            artifact.meta.usize_field("n_reg")?,
        );
        if d_model == 0 || n_tokens == 0 {
            return Err("artifact records a zero d_model or n_tokens".into());
        }
        if n_reg >= n_tokens {
            return Err(format!(
                "n_reg {n_reg} leaves no token to pool out of {n_tokens}"
            ));
        }

        let tokenizer =
            tokenizer_from_artifact(artifact, seq_len, in_channels, d_model, n_tokens, n_reg)?;
        validate_pos_embed(artifact, tokenizer, n_tokens, d_model)?;

        let input_proj = artifact.meta.bool_field("input_proj")?;
        let has_input_proj = artifact.shape_of("encoder.input_proj.weight").is_some();
        if input_proj != has_input_proj {
            return Err(format!(
                "artifact records input_proj = {input_proj} but encoder.input_proj.weight is \
                 {}",
                if has_input_proj { "present" } else { "missing" }
            ));
        }
        if input_proj {
            let shape = artifact
                .shape_of("encoder.input_proj.weight")
                .ok_or("missing encoder.input_proj.weight")?;
            if shape.as_slice() != [d_model, d_model] {
                return Err(format!(
                    "encoder.input_proj.weight {shape:?} is not a {d_model} x {d_model} projection"
                ));
            }
        }

        let tail = tail_from_artifact(artifact, d_model)?;

        let n_heads = artifact.meta.usize_field("nhead")?;
        if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(format!("nhead {n_heads} does not divide d_model {d_model}"));
        }
        // Read the layer count rather than probing until a lookup fails. Probing
        // silently accepts a gap in the numbering (layers 0, 1, 3 would count as
        // two) and allocates a formatted key per attempt.
        let num_layers = artifact.meta.usize_field("num_layers")?;
        if num_layers == 0 {
            return Err("artifact records num_layers = 0".into());
        }
        for l in 0..num_layers {
            let key = format!("encoder.transformer.layers.{l}.norm1.weight");
            if artifact.shape_of(&key).is_none() {
                return Err(format!("artifact is missing {key}"));
            }
        }

        // head.3 is the final classification projection: (num_classes, hidden).
        let num_classes = *artifact
            .shape_of("head.3.weight")
            .and_then(|s| s.first())
            .ok_or("missing head.3.weight")?;

        Ok(Self {
            seq_len,
            in_channels,
            d_model,
            n_tokens,
            n_reg,
            n_heads,
            num_layers,
            num_classes,
            tokenizer,
            tail,
            input_proj,
        })
    }
}

/// The tokenizer weights, tagged with the branch they implement.
///
/// Built from a validated [`Config`], so the variant and the tensors cannot
/// disagree.
enum TokenizerWeights {
    Linear {
        weight: PackedWeight,
        bias: Tensor,
    },
    Patch {
        stride: usize,
        weight: Tensor,
        bias: Tensor,
    },
    Feature {
        n_reg: usize,
        total_features: usize,
        weight: Tensor,
        bias: Tensor,
        feat_idx: Tensor,
    },
}

/// A loaded world-model classifier, holding dequantized weights.
pub struct Model {
    /// Shape parameters, validated against the loaded weights.
    pub config: Config,
    tokenizer: TokenizerWeights,
    pos_embed: Option<Tensor>,
    input_proj: Option<(PackedWeight, Tensor)>,
    layers: Vec<Layer>,
    tail: Tail,
    attn_w: PackedWeight,
    attn_b: Tensor,
    head0_w: PackedWeight,
    head0_b: Tensor,
    head3_w: PackedWeight,
    head3_b: Tensor,
    eps_ln: f32,
    eps_bn: f32,
}

struct Layer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    q_w: PackedWeight,
    q_b: Tensor,
    k_w: PackedWeight,
    k_b: Tensor,
    v_w: PackedWeight,
    v_b: Tensor,
    out_w: PackedWeight,
    out_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    ff1_w: PackedWeight,
    ff1_b: Tensor,
    ff2_w: PackedWeight,
    ff2_b: Tensor,
}

enum Tail {
    LayerNorm {
        weight: Tensor,
        bias: Tensor,
    },
    ProjectBatchNorm {
        proj_w: PackedWeight,
        proj_b: Tensor,
        bn_w: Tensor,
        bn_b: Tensor,
        bn_mean: Tensor,
        bn_var: Tensor,
    },
}

/// Prepend `n_reg` zero rows to a `(features, d_model)` tensor.
///
/// `T-JEPA`'s `[REG]` slots have no bias and no feature-index embedding: the
/// reference concatenates `torch.zeros` at forward time, so the padding is
/// materialised once here instead of branched on per element.
fn pad_rows(x: &Tensor, n_reg: usize, d_model: usize) -> Tensor {
    if n_reg == 0 {
        return x.clone();
    }
    let rows = x.data.len() / d_model;
    let mut data = vec![0.0f32; n_reg * d_model];
    data.extend_from_slice(&x.data);
    Tensor::new(data, vec![n_reg + rows, d_model])
}

/// `T-JEPA`'s tokenizer.
///
/// `n_reg` slots whose input value is the constant 1.0 are prepended to the
/// window flattened timestep-major (`torch`'s `x.reshape(B, T*C)`), then every
/// slot is scaled elementwise by its own row of `weight`:
///
/// ```text
/// out[i, d] = weight[i, d] * x[i] + bias[i, d] + feat_idx[i, d]
/// ```
///
/// `bias` and `feat_idx` arrive already padded with `n_reg` zero rows.
fn feature_tokenize(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    feat_idx: &Tensor,
    n_reg: usize,
    total_features: usize,
    d_model: usize,
) -> Tensor {
    let n_tokens = n_reg + total_features;
    let mut out = vec![0.0f32; n_tokens * d_model];
    for i in 0..n_tokens {
        // The [REG] slots are fed the constant 1.0; `x` supplies the rest.
        let value = if i < n_reg { 1.0 } else { x.data[i - n_reg] };
        for d in 0..d_model {
            let j = i * d_model + d;
            out[j] = weight.data[j] * value + bias.data[j] + feat_idx.data[j];
        }
    }
    Tensor::new(out, vec![n_tokens, d_model])
}

impl Model {
    /// Load every weight from the artifact, dequantizing as needed.
    pub fn load(artifact: &Artifact) -> Result<Self, String> {
        let config = Config::from_artifact(artifact)?;
        let t = |name: &str| artifact.tensor(name);
        // Every weight that feeds a `linear` is transposed once, here, so the
        // forward pass can use the vectorisable kernel. See `PackedWeight`.
        let tw = |name: &str| artifact.tensor(name).map(|w| PackedWeight::pack(&w));
        let d_model = config.d_model;

        let tokenizer = match config.tokenizer {
            Tokenizer::Linear => TokenizerWeights::Linear {
                weight: tw("encoder.tokenizer.weight")?,
                bias: t("encoder.tokenizer.bias")?,
            },
            Tokenizer::Patch { stride, .. } => TokenizerWeights::Patch {
                stride,
                weight: t("encoder.tokenizer.proj.weight")?,
                bias: t("encoder.tokenizer.proj.bias")?,
            },
            Tokenizer::Feature { total_features } => TokenizerWeights::Feature {
                n_reg: config.n_reg,
                total_features,
                // `encoder.tokenizer.reg_tokens.tokens` is in the state dict but
                // `FeatureTokenizer.forward` never reads it — the [REG] input
                // value is the constant 1.0 — so it is deliberately not loaded.
                weight: t("encoder.tokenizer.weight")?,
                bias: pad_rows(&t("encoder.tokenizer.bias")?, config.n_reg, d_model),
                feat_idx: pad_rows(
                    &t("encoder.tokenizer.feat_idx_embed.weight")?,
                    config.n_reg,
                    d_model,
                ),
            },
        };

        let pos_embed = match artifact.shape_of("encoder.pos_embed") {
            Some(_) => Some(t("encoder.pos_embed")?),
            None => None,
        };

        let input_proj = if config.input_proj {
            Some((
                tw("encoder.input_proj.weight")?,
                t("encoder.input_proj.bias")?,
            ))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let p = format!("encoder.transformer.layers.{l}");
            layers.push(Layer {
                norm1_w: t(&format!("{p}.norm1.weight"))?,
                norm1_b: t(&format!("{p}.norm1.bias"))?,
                q_w: tw(&format!("{p}.self_attn.q.weight"))?,
                q_b: t(&format!("{p}.self_attn.q.bias"))?,
                k_w: tw(&format!("{p}.self_attn.k.weight"))?,
                k_b: t(&format!("{p}.self_attn.k.bias"))?,
                v_w: tw(&format!("{p}.self_attn.v.weight"))?,
                v_b: t(&format!("{p}.self_attn.v.bias"))?,
                out_w: tw(&format!("{p}.self_attn.out_proj.weight"))?,
                out_b: t(&format!("{p}.self_attn.out_proj.bias"))?,
                norm2_w: t(&format!("{p}.norm2.weight"))?,
                norm2_b: t(&format!("{p}.norm2.bias"))?,
                ff1_w: tw(&format!("{p}.linear1.weight"))?,
                ff1_b: t(&format!("{p}.linear1.bias"))?,
                ff2_w: tw(&format!("{p}.linear2.weight"))?,
                ff2_b: t(&format!("{p}.linear2.bias"))?,
            });
        }

        let tail = match config.tail {
            EncoderTail::LayerNorm => Tail::LayerNorm {
                weight: t("encoder.norm.weight")?,
                bias: t("encoder.norm.bias")?,
            },
            EncoderTail::ProjectBatchNorm => Tail::ProjectBatchNorm {
                proj_w: tw("encoder.proj.0.weight")?,
                proj_b: t("encoder.proj.0.bias")?,
                bn_w: t("encoder.bn.weight")?,
                bn_b: t("encoder.bn.bias")?,
                bn_mean: t("encoder.bn.running_mean")?,
                bn_var: t("encoder.bn.running_var")?,
            },
        };

        Ok(Self {
            eps_ln: artifact.meta.eps.layernorm,
            eps_bn: artifact.meta.eps.batchnorm,
            config,
            tokenizer,
            pos_embed,
            input_proj,
            layers,
            tail,
            attn_w: tw("attn.weight")?,
            attn_b: t("attn.bias")?,
            head0_w: tw("head.0.weight")?,
            head0_b: t("head.0.bias")?,
            head3_w: tw("head.3.weight")?,
            head3_b: t("head.3.bias")?,
        })
    }

    /// Tokenize a `(seq_len, in_channels)` window into `(n_tokens, d_model)`.
    fn tokenize(&self, x: &Tensor) -> Tensor {
        match &self.tokenizer {
            TokenizerWeights::Linear { weight, bias } => ops::linear_packed(x, weight, bias),
            TokenizerWeights::Patch {
                stride,
                weight,
                bias,
            } => {
                // Conv1d convolves over (channels, length), so the channel axis
                // has to move in front of time and back again. Reinterpreting
                // the same memory as (length, channels) would reorder elements.
                let channels_first = ops::transpose_2d(x);
                let patches = ops::conv1d(&channels_first, weight, bias, *stride);
                ops::transpose_2d(&patches)
            }
            TokenizerWeights::Feature {
                n_reg,
                total_features,
                weight,
                bias,
                feat_idx,
            } => feature_tokenize(
                x,
                weight,
                bias,
                feat_idx,
                *n_reg,
                *total_features,
                self.config.d_model,
            ),
        }
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

        let d_model = self.config.d_model;
        let n_tokens = self.config.n_tokens;
        let n_reg = self.config.n_reg;

        let mut z = self.tokenize(x);
        assert_eq!(
            z.shape,
            vec![n_tokens, d_model],
            "tokenizer produced the wrong shape"
        );
        if let Some(pos_embed) = &self.pos_embed {
            z = ops::add(&z, pos_embed);
        }
        if let Some((weight, bias)) = &self.input_proj {
            z = ops::linear_packed(&z, weight, bias);
        }

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
            let ff = ops::linear_packed(&normed, &layer.ff1_w, &layer.ff1_b);
            let ff = ops::gelu(&ff);
            let ff = ops::linear_packed(&ff, &layer.ff2_w, &layer.ff2_b);
            z = ops::add(&z, &ff);
        }

        let z = match &self.tail {
            Tail::LayerNorm { weight, bias } => ops::layer_norm(&z, weight, bias, self.eps_ln),
            Tail::ProjectBatchNorm {
                proj_w,
                proj_b,
                bn_w,
                bn_b,
                bn_mean,
                bn_var,
            } => {
                let proj = ops::linear_packed(&z, proj_w, proj_b);
                let proj = ops::gelu(&proj);
                // BatchNorm1d normalises over the width axis, so the (tokens,
                // width) layout is transposed for it and transposed back.
                let transposed = ops::transpose_2d(&proj);
                let normed =
                    ops::batch_norm_eval(&transposed, bn_w, bn_b, bn_mean, bn_var, self.eps_bn);
                ops::transpose_2d(&normed)
            }
        };

        // Attention pooling over the features only. `T-JEPA`'s [REG] slots are a
        // training-time pressure valve and `TJEPAClassifier.forward` drops them
        // here, before the pooling scores are computed.
        let tokens = n_tokens - n_reg;
        let scores = ops::linear_packed(&z, &self.attn_w, &self.attn_b);
        let scores = if n_reg == 0 {
            scores
        } else {
            Tensor::new(scores.data[n_reg..].to_vec(), vec![tokens, 1])
        };
        let weights = ops::softmax_first_axis(&scores);
        let mut pooled = vec![0.0f32; d_model];
        for i in 0..tokens {
            let weight = weights.data[i];
            let row = &z.data[(i + n_reg) * d_model..(i + n_reg + 1) * d_model];
            for (acc, value) in pooled.iter_mut().zip(row) {
                *acc += value * weight;
            }
        }
        let pooled = Tensor::new(pooled, vec![1, d_model]);

        // classification head
        let hidden = ops::linear_packed(&pooled, &self.head0_w, &self.head0_b);
        let hidden = ops::gelu(&hidden);
        ops::linear_packed(&hidden, &self.head3_w, &self.head3_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_stride_recovers_the_conv_configuration() {
        // TS-JEPA and CF-JEPA: seq_len 10, kernel/stride 2, 5 patches.
        assert_eq!(patch_stride(10, 2, 5).unwrap(), 2);
        // A stride-1 overlapping tokenizer must not be read as stride == kernel.
        assert_eq!(patch_stride(10, 2, 9).unwrap(), 1);
        assert_eq!(patch_stride(12, 3, 4).unwrap(), 3);
    }

    #[test]
    fn patch_stride_rejects_a_patch_count_the_kernel_cannot_produce() {
        // 10 timesteps with kernel 2 cannot yield 4 patches at any integer stride.
        assert!(patch_stride(10, 2, 4).is_err());
        // Kernel longer than the window, and a degenerate single token.
        assert!(patch_stride(1, 2, 1).is_err());
        assert!(patch_stride(10, 2, 1).is_err());
    }

    #[test]
    fn feature_tokenizer_scales_then_adds_bias_and_index_embedding() {
        // 1 [REG] slot + 2 features, width 2. Slot i is weight[i]*x[i] +
        // bias[i] + feat_idx[i], with the [REG] input value being 1.0.
        let x = Tensor::new(vec![3.0, 4.0], vec![2, 1]);
        let weight = Tensor::new(vec![101.0, 99.0, 1.0, 0.0, 2.0, 0.0], vec![3, 2]);
        let bias = Tensor::new(vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0], vec![3, 2]);
        let feat_idx = Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 2.0, 8.0], vec![3, 2]);

        let got = feature_tokenize(&x, &weight, &bias, &feat_idx, 1, 2, 2);

        assert_eq!(got.shape, vec![3, 2]);
        // [REG]: 1.0 * weight + 0 + 0
        // feat0: 3 * [1,0] + [0,5] + [0,0]
        // feat1: 4 * [2,0] + [0,0] + [2,8]
        assert_eq!(got.data, vec![101.0, 99.0, 3.0, 5.0, 10.0, 8.0]);
    }

    #[test]
    fn reg_rows_of_the_bias_and_index_embedding_are_zero() {
        // The exporter stores bias/feat_idx for the features only; the [REG]
        // rows are the zero padding the reference concatenates.
        let features = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let padded = pad_rows(&features, 2, 2);
        assert_eq!(padded.shape, vec![4, 2]);
        assert_eq!(padded.data, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        // No [REG] slots means no padding at all.
        assert_eq!(pad_rows(&features, 0, 2).data, features.data);
    }

    /// A minimal T-JEPA model, built by hand so the tokenizer and the pooling
    /// rule can be exercised without exporting a checkpoint.
    ///
    /// Width 2, one [REG] slot, two features, no transformer layers, identity
    /// head: the logits are `gelu(pooled)`, so the pooled vector is readable.
    fn mini_t_jepa() -> Model {
        let ones = Tensor::new(vec![1.0, 1.0], vec![1, 2]);
        Model {
            config: Config {
                seq_len: 2,
                in_channels: 1,
                d_model: 2,
                n_tokens: 3,
                n_reg: 1,
                n_heads: 1,
                num_layers: 0,
                num_classes: 2,
                tokenizer: Tokenizer::Feature { total_features: 2 },
                tail: EncoderTail::LayerNorm,
                input_proj: false,
            },
            tokenizer: TokenizerWeights::Feature {
                n_reg: 1,
                total_features: 2,
                // Rows evaluate to [101,99] ([REG]), [3,5] and [10,8] for x=[3,4].
                weight: Tensor::new(vec![101.0, 99.0, 1.0, 0.0, 2.0, 0.0], vec![3, 2]),
                bias: Tensor::new(vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0], vec![3, 2]),
                feat_idx: Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 2.0, 8.0], vec![3, 2]),
            },
            pos_embed: None,
            input_proj: None,
            layers: Vec::new(),
            tail: Tail::LayerNorm {
                weight: ones.clone(),
                bias: Tensor::new(vec![0.0, 0.0], vec![1, 2]),
            },
            // attn reads the first component only.
            attn_w: PackedWeight::pack(&Tensor::new(vec![1.0, 0.0], vec![1, 2])),
            attn_b: Tensor::new(vec![0.0], vec![1]),
            head0_w: PackedWeight::pack(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])),
            head0_b: Tensor::new(vec![0.0, 0.0], vec![2]),
            head3_w: PackedWeight::pack(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])),
            head3_b: Tensor::new(vec![0.0, 0.0], vec![2]),
            eps_ln: 0.0,
            eps_bn: 0.0,
        }
    }

    #[test]
    fn pooling_drops_the_reg_tokens() {
        // LayerNorm(eps 0) maps the three rows to [1,-1], [-1,1], [1,-1].
        // With the [REG] row included the pooling scores would be [-1, 1, -1]
        // and the softmax would put 46.8% on [1,-1] and 46.8% on the [REG] row.
        // Dropping the [REG] row leaves scores [-1, 1], whose softmax weights
        // are 0.119 and 0.881 -> pooled = [tanh(1), -tanh(1)].
        let model = mini_t_jepa();
        let x = Tensor::new(vec![3.0, 4.0], vec![2, 1]);
        let logits = model.forward(&x);

        let t = libm::tanhf(1.0);
        let expected = ops::gelu(&Tensor::new(vec![t, -t], vec![1, 2]));
        assert!(
            logits.max_abs_diff(&expected) < 1e-5,
            "pooling did not drop the [REG] slot: got {:?}, expected {t}",
            logits.data
        );
    }
}

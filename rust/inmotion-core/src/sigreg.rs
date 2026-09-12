//! The `SIGReg` classifier: a multi-scale CNN, not a transformer.
//!
//! `SIGReg` is grouped with the world models because it shares their Gaussian
//! latent regulariser at training time, but its inference graph is an ordinary
//! convolutional network. Forcing it through [`crate::world_model`] would mean
//! inventing a tokenizer and an attention pool it does not have, so it is its own
//! module sharing only [`crate::ops`].
//!
//! Mirrors `sigreg_classifier.py` exactly:
//!
//! ```text
//! x = cat([conv3(x), conv5(x), conv7(x)], dim = channel)   # padding = k / 2
//! x = gelu(batch_norm(x))
//! for each residual block:                                # num_blocks
//!     h = batch_norm(conv(x))
//!     h = gelu(h)
//!     h = batch_norm(gelu(conv(h)))
//!     x = gelu(x + h)
//! f = mean(x, dim = time)                                  # AdaptiveAvgPool1d(1)
//! l = layer_norm(latent_proj(f))
//! logits = classifier(l)
//! ```
//!
//! Both `BatchNorm1d` layers of every block, and the one after the multi-scale
//! front end, are folded into per-channel `(scale, shift)` pairs at load time, so
//! inference never touches a running variance.

use crate::artifact::Artifact;
use crate::ops;
use crate::tensor::Tensor;

/// Architecture hyperparameters, read from the artifact.
#[derive(Debug, Clone)]
pub struct Config {
    /// Timesteps per window.
    pub seq_len: usize,
    /// Input channels per timestep.
    pub in_features: usize,
    /// Width of the CNN feature maps; also the input of `latent_proj`.
    pub num_filters: usize,
    /// Number of residual blocks.
    pub num_blocks: usize,
    /// Width of the latent bottleneck.
    pub latent_dim: usize,
    /// Output classes.
    pub num_classes: usize,
}

/// The kernels `_CNNEncoder` uses for its multi-scale front end.
const FRONT_KERNELS: [usize; 3] = [3, 5, 7];
/// The kernels `_ResBlock` uses for both of its convolutions.
const BLOCK_KERNELS: [usize; 2] = [3, 3];

impl Config {
    /// Derive the shape parameters from the artifact's tensor index.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, String> {
        let family = artifact.meta.str_field("family")?;
        if family != "sigreg_cnn" {
            return Err(format!(
                "expected family `sigreg_cnn`, artifact says {family:?}"
            ));
        }
        let seq_len = artifact.meta.seq_len;
        let in_features = artifact.meta.in_channels;
        let num_filters = artifact.meta.usize_field("num_filters")?;
        let num_blocks = artifact.meta.usize_field("num_blocks")?;
        if seq_len == 0 || num_filters == 0 || num_blocks == 0 {
            return Err("artifact records a zero seq_len, num_filters or num_blocks".into());
        }

        // The kernels are part of the architecture, not a property of the
        // weights, so they are declared by the artifact and checked here rather
        // than silently assumed.
        let front_kernels = artifact.meta.usize_list_field("front_kernels")?;
        if front_kernels != FRONT_KERNELS {
            return Err(format!(
                "artifact declares front_kernels {front_kernels:?}, this port implements \
                 {FRONT_KERNELS:?}"
            ));
        }
        let block_kernels = artifact.meta.usize_list_field("block_kernels")?;
        if block_kernels != BLOCK_KERNELS {
            return Err(format!(
                "artifact declares block_kernels {block_kernels:?}, this port implements \
                 {BLOCK_KERNELS:?}"
            ));
        }

        // The front end's widths follow the reference's split of `num_filters`
        // into thirds, with the 7-wide branch taking the remainder.
        let third = num_filters / 3;
        let expected_out = [third, third, num_filters - 2 * third];
        for (i, kernel) in FRONT_KERNELS.iter().enumerate() {
            let name = format!("encoder.conv{kernel}.weight");
            let shape = artifact
                .shape_of(&name)
                .ok_or_else(|| format!("missing {name}"))?;
            if shape.as_slice() != [expected_out[i], in_features, *kernel] {
                return Err(format!(
                    "{name} {shape:?} is not ({} x {in_features} x {kernel})",
                    expected_out[i]
                ));
            }
        }

        let bn = artifact
            .shape_of("encoder.bn.weight")
            .ok_or("missing encoder.bn.weight")?;
        if bn.as_slice() != [num_filters] {
            return Err(format!(
                "encoder.bn.weight {bn:?} does not normalise {num_filters} channels"
            ));
        }

        for block in 0..num_blocks {
            for (index, kernel) in BLOCK_KERNELS.iter().enumerate() {
                let step = if index == 0 { 0 } else { 3 };
                let name = format!("encoder.blocks.{block}.block.{step}.weight");
                let shape = artifact
                    .shape_of(&name)
                    .ok_or_else(|| format!("missing {name}"))?;
                if shape.as_slice() != [num_filters, num_filters, *kernel] {
                    return Err(format!(
                        "{name} {shape:?} is not ({num_filters} x {num_filters} x {kernel})"
                    ));
                }
            }
        }

        let proj = artifact
            .shape_of("latent_proj.weight")
            .ok_or("missing latent_proj.weight")?;
        if proj.len() != 2 || proj[1] != num_filters {
            return Err(format!(
                "latent_proj.weight {proj:?} does not project the {num_filters} pooled features"
            ));
        }
        let latent_dim = proj[0];

        let classifier = artifact
            .shape_of("classifier.weight")
            .ok_or("missing classifier.weight")?;
        if classifier.len() != 2 || classifier[1] != latent_dim {
            return Err(format!(
                "classifier.weight {classifier:?} does not consume {latent_dim} latents"
            ));
        }
        let num_classes = classifier[0];

        Ok(Self {
            seq_len,
            in_features,
            num_filters,
            num_blocks,
            latent_dim,
            num_classes,
        })
    }
}

/// A folded `BatchNorm1d`: the per-channel affine it applies in eval mode.
struct Affine {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl Affine {
    /// Fold `{prefix}.weight/bias/running_mean/running_var` once, at load.
    fn load(artifact: &Artifact, prefix: &str) -> Result<Self, String> {
        let weight = artifact.tensor(&format!("{prefix}.weight"))?;
        let bias = artifact.tensor(&format!("{prefix}.bias"))?;
        let mean = artifact.tensor(&format!("{prefix}.running_mean"))?;
        let var = artifact.tensor(&format!("{prefix}.running_var"))?;
        let (scale, shift) =
            ops::fold_batch_norm(&weight, &bias, &mean, &var, artifact.meta.eps.batchnorm);
        Ok(Self { scale, shift })
    }

    fn apply(&self, x: &Tensor) -> Tensor {
        ops::channel_affine(x, &self.scale, &self.shift)
    }
}

/// One `Conv1d`, holding its kernel width so callers need not re-read the shape.
struct Conv {
    weight: Tensor,
    bias: Tensor,
}

impl Conv {
    fn load(artifact: &Artifact, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: artifact.tensor(&format!("{prefix}.weight"))?,
            bias: artifact.tensor(&format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        ops::conv1d_same(x, &self.weight, &self.bias)
    }
}

/// `_ResBlock`: `gelu(x + batch_norm(conv(gelu(batch_norm(conv(x))))))`.
struct Block {
    conv1: Conv,
    bn1: Affine,
    conv2: Conv,
    bn2: Affine,
}

impl Block {
    fn load(artifact: &Artifact, index: usize) -> Result<Self, String> {
        let p = format!("encoder.blocks.{index}.block");
        Ok(Self {
            conv1: Conv::load(artifact, &format!("{p}.0"))?,
            bn1: Affine::load(artifact, &format!("{p}.1"))?,
            conv2: Conv::load(artifact, &format!("{p}.3"))?,
            bn2: Affine::load(artifact, &format!("{p}.4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = self.conv1.forward(x);
        let hidden = self.bn1.apply(&hidden);
        let hidden = ops::gelu(&hidden);
        let hidden = self.conv2.forward(&hidden);
        let hidden = self.bn2.apply(&hidden);
        ops::gelu(&ops::add(x, &hidden))
    }
}

/// A loaded `SIGReg` classifier, holding dequantized weights.
pub struct Model {
    /// Shape parameters, validated against the loaded weights.
    pub config: Config,
    front: Vec<Conv>,
    front_bn: Affine,
    blocks: Vec<Block>,
    latent_w: Tensor,
    latent_b: Tensor,
    latent_norm_w: Tensor,
    latent_norm_b: Tensor,
    classifier_w: Tensor,
    classifier_b: Tensor,
    eps_ln: f32,
}

impl Model {
    /// Load every weight from the artifact, folding the normalisations.
    pub fn load(artifact: &Artifact) -> Result<Self, String> {
        let config = Config::from_artifact(artifact)?;

        let mut front = Vec::with_capacity(FRONT_KERNELS.len());
        for kernel in FRONT_KERNELS {
            front.push(Conv::load(artifact, &format!("encoder.conv{kernel}"))?);
        }
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for index in 0..config.num_blocks {
            blocks.push(Block::load(artifact, index)?);
        }

        Ok(Self {
            eps_ln: artifact.meta.eps.layernorm,
            front_bn: Affine::load(artifact, "encoder.bn")?,
            front,
            blocks,
            latent_w: artifact.tensor("latent_proj.weight")?,
            latent_b: artifact.tensor("latent_proj.bias")?,
            latent_norm_w: artifact.tensor("latent_norm.weight")?,
            latent_norm_b: artifact.tensor("latent_norm.bias")?,
            classifier_w: artifact.tensor("classifier.weight")?,
            classifier_b: artifact.tensor("classifier.bias")?,
            config,
        })
    }

    /// Run one window, `x: (seq_len, in_features)`; returns `(1, num_classes)` logits.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (t, c) = x.as_2d();
        assert_eq!(
            c, self.config.in_features,
            "expected {} input channels",
            self.config.in_features
        );
        assert_eq!(
            t, self.config.seq_len,
            "expected {} timesteps",
            self.config.seq_len
        );

        // The convolutions run over (channels, length), so the channel axis has
        // to move in front of time. Reinterpreting the same memory would reorder
        // every element.
        let channels_first = ops::transpose_2d(x);
        let branches: Vec<Tensor> = self
            .front
            .iter()
            .map(|conv| conv.forward(&channels_first))
            .collect();
        let mut z = ops::concat_channels(&branches);
        z = self.front_bn.apply(&z);
        z = ops::gelu(&z);

        for block in &self.blocks {
            z = block.forward(&z);
        }

        // AdaptiveAvgPool1d(1): the mean over time, per channel.
        let length = z.shape[1];
        let mut pooled = vec![0.0f32; self.config.num_filters];
        for (channel, value) in pooled.iter_mut().enumerate() {
            let row = &z.data[channel * length..(channel + 1) * length];
            *value = row.iter().sum::<f32>() / length as f32;
        }
        let features = Tensor::new(pooled, vec![1, self.config.num_filters]);

        // Latent bottleneck, then the classifier head. The latents are not
        // returned: only the logits leave the device.
        let latents = ops::linear(&features, &self.latent_w, &self.latent_b);
        let latents = ops::layer_norm(
            &latents,
            &self.latent_norm_w,
            &self.latent_norm_b,
            self.eps_ln,
        );
        ops::linear(&latents, &self.classifier_w, &self.classifier_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_folds_the_running_statistics_into_a_scale_and_shift() {
        // weight 2, bias 1, mean 3, var 4, eps 0 -> scale 1, shift -2.
        let weight = Tensor::new(vec![2.0], vec![1]);
        let bias = Tensor::new(vec![1.0], vec![1]);
        let mean = Tensor::new(vec![3.0], vec![1]);
        let var = Tensor::new(vec![4.0], vec![1]);
        let (scale, shift) = ops::fold_batch_norm(&weight, &bias, &mean, &var, 0.0);
        let affine = Affine { scale, shift };

        let x = Tensor::new(vec![1.0, 3.0, 7.0], vec![1, 3]);
        // (x - 3) / 2 * 2 + 1 = x - 2
        assert_eq!(affine.apply(&x).data, vec![-1.0, 1.0, 5.0]);
    }

    #[test]
    fn a_residual_block_is_gelu_of_the_input_plus_its_inner_path() {
        // One channel, kernel 3 (padding 1), all-ones weights, no bias, identity
        // normalisations: the inner path is the twice-convolved signal.
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]);
        let bias = Tensor::new(vec![0.0], vec![1]);
        let block = Block {
            conv1: Conv {
                weight: weight.clone(),
                bias: bias.clone(),
            },
            bn1: Affine {
                scale: vec![1.0],
                shift: vec![0.0],
            },
            conv2: Conv { weight, bias },
            bn2: Affine {
                scale: vec![1.0],
                shift: vec![0.0],
            },
        };

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let got = block.forward(&x);
        assert_eq!(got.shape, vec![1, 4]);

        // Hand-computed, without reusing `ops::gelu`, so this pins the wiring:
        // conv over [1,2,3,4] is [3,6,9,7]; the second conv sums three of those.
        let gelu = |v: f32| 0.5 * v * (1.0 + libm::erff(v / 2.0f32.sqrt()));
        let g = [3.0f32, 6.0, 9.0, 7.0].map(gelu);
        let inner = [
            g[0] + g[1],
            g[0] + g[1] + g[2],
            g[1] + g[2] + g[3],
            g[2] + g[3],
        ];
        let expected = [
            gelu(1.0 + inner[0]),
            gelu(2.0 + inner[1]),
            gelu(3.0 + inner[2]),
            gelu(4.0 + inner[3]),
        ];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (got.data[i] - want).abs() < 1e-5,
                "residual block output {i}: got {}, want {want}",
                got.data[i]
            );
        }
    }
}

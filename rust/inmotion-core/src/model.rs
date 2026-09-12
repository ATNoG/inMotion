//! Family dispatch: the artifact says which model it holds.
//!
//! Two graphs live in this crate — the transformer world models and the `SIGReg`
//! CNN. The exporter records `arch.family`, so callers ask [`Model::load`] for a
//! model and never branch on the model's name themselves.

use crate::artifact::Artifact;
use crate::sigreg;
use crate::tensor::Tensor;
use crate::world_model;

/// The model families this runtime implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Pre-norm transformer, attention pooling: `lejepa`, `t_jepa`, `ts_jepa`,
    /// `cf_jepa`.
    Transformer,
    /// Multi-scale CNN with a Gaussian latent bottleneck: `sigreg`.
    SigRegCnn,
}

impl Family {
    /// Read the family the exporter recorded.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, String> {
        match artifact.meta.str_field("family")? {
            "transformer" => Ok(Self::Transformer),
            "sigreg_cnn" => Ok(Self::SigRegCnn),
            other => Err(format!("unknown model family {other:?}")),
        }
    }

    /// The name the exporter writes for this family.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::SigRegCnn => "sigreg_cnn",
        }
    }
}

/// A loaded model of any family.
/// Both variants are boxed: a model is loaded once and lives for the whole
/// process, so moving it behind a pointer costs nothing and keeps the enum small.
pub enum Model {
    /// One of the transformer world models.
    Transformer(Box<world_model::Model>),
    /// The `SIGReg` CNN.
    SigReg(Box<sigreg::Model>),
}

impl Model {
    /// Load whichever model the artifact holds.
    pub fn load(artifact: &Artifact) -> Result<Self, String> {
        match Family::from_artifact(artifact)? {
            Family::Transformer => Ok(Self::Transformer(Box::new(world_model::Model::load(
                artifact,
            )?))),
            Family::SigRegCnn => Ok(Self::SigReg(Box::new(sigreg::Model::load(artifact)?))),
        }
    }

    /// The family this model belongs to.
    #[must_use]
    pub fn family(&self) -> Family {
        match self {
            Self::Transformer(_) => Family::Transformer,
            Self::SigReg(_) => Family::SigRegCnn,
        }
    }

    /// Timesteps per window.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        match self {
            Self::Transformer(m) => m.config.seq_len,
            Self::SigReg(m) => m.config.seq_len,
        }
    }

    /// Input channels per timestep.
    #[must_use]
    pub fn in_channels(&self) -> usize {
        match self {
            Self::Transformer(m) => m.config.in_channels,
            Self::SigReg(m) => m.config.in_features,
        }
    }

    /// Output classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        match self {
            Self::Transformer(m) => m.config.num_classes,
            Self::SigReg(m) => m.config.num_classes,
        }
    }

    /// Run one window, `x: (seq_len, in_channels)`; returns `(1, num_classes)`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Self::Transformer(m) => m.forward(x),
            Self::SigReg(m) => m.forward(x),
        }
    }

    /// Architecture lines for `inmotion-infer describe`.
    ///
    /// Each family reports the fields that actually shape its graph, so a CNN is
    /// never described with transformer hyperparameters it does not have.
    #[must_use]
    pub fn describe(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Transformer(m) => {
                let c = &m.config;
                vec![
                    ("d_model", c.d_model.to_string()),
                    ("n_heads", c.n_heads.to_string()),
                    ("num_layers", c.num_layers.to_string()),
                    ("n_tokens", c.n_tokens.to_string()),
                    ("n_reg", c.n_reg.to_string()),
                    (
                        "tokenizer",
                        match c.tokenizer {
                            world_model::Tokenizer::Linear => "linear (per timestep)".to_string(),
                            world_model::Tokenizer::Patch { kernel, stride } => {
                                format!("patch conv1d(kernel {kernel}, stride {stride})")
                            }
                            world_model::Tokenizer::Feature { total_features } => format!(
                                "feature ({total_features} per-slot scales + feature-index embedding)"
                            ),
                        },
                    ),
                    (
                        "tail",
                        match c.tail {
                            world_model::EncoderTail::LayerNorm => "layernorm".to_string(),
                            world_model::EncoderTail::ProjectBatchNorm => {
                                "proj + gelu + batchnorm".to_string()
                            }
                        },
                    ),
                ]
            }
            Self::SigReg(m) => {
                let c = &m.config;
                vec![
                    ("num_filters", c.num_filters.to_string()),
                    ("num_blocks", c.num_blocks.to_string()),
                    ("latent_dim", c.latent_dim.to_string()),
                ]
            }
        }
    }
}

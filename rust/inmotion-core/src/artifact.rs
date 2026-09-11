//! Reading the exported artifact: a safetensors weight bag plus `meta.json`.
//!
//! The safetensors reader is written here rather than pulled in as a
//! dependency. The format is a little-endian `u64` header length, a JSON header,
//! then the raw buffers, which is short enough to audit and keeps the dependency
//! set small enough for a microcontroller build.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::quant::{Granularity, QuantizedWeight};
use crate::tensor::Tensor;

/// One entry in the artifact's tensor index.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    /// True when the weights were stored as packed integer codes.
    pub quantized: bool,
    /// Shape of the original tensor, before any quantization.
    pub shape: Vec<usize>,
    /// Code width, present only when `quantized`.
    #[serde(default)]
    pub bits: Option<u32>,
    /// `per_tensor`, `per_channel` or `per_group`, when `quantized`.
    #[serde(default)]
    pub granularity: Option<String>,
    /// Elements per group for `per_group`.
    #[serde(default)]
    pub group_size: Option<usize>,
    /// Axis for `per_channel`.
    #[serde(default)]
    pub axis: Option<usize>,
    /// Shape of the scale metadata, for cross-checking on load.
    #[serde(default)]
    pub scale_shape: Option<Vec<usize>>,
}

/// The `meta.json` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    /// Format version; the loader refuses anything it does not know.
    pub artifact_version: u32,
    /// Registry name of the model, e.g. `lejepa`.
    pub model: String,
    /// Factory that built it, for traceability back to the registry.
    pub factory: String,
    /// Whether the model consumes the 18-channel feature expansion.
    pub rich_features: bool,
    /// Timesteps per inference window.
    pub seq_len: usize,
    /// Input channels per timestep (4 raw, 18 rich).
    pub in_channels: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Normalization epsilons, taken from the artifact rather than assumed.
    pub eps: Eps,
    /// Architecture hyperparameters as exported (`d_model`, nhead, `num_layers`, ...).
    #[serde(default)]
    pub arch: HashMap<String, serde_json::Value>,
    /// Every tensor in the artifact, keyed by its name in the weights file.
    pub tensors: HashMap<String, TensorInfo>,
    /// The quantization scheme used, or `None` for a plain float export.
    #[serde(default)]
    pub quant: Option<serde_json::Value>,
}

/// Normalization epsilons, taken from the artifact rather than hardcoded.
#[derive(Debug, Clone, Deserialize)]
pub struct Eps {
    /// Epsilon used by `nn.LayerNorm`.
    pub layernorm: f32,
    /// Epsilon used by `nn.BatchNorm1d`.
    pub batchnorm: f32,
}

/// A loaded artifact.
///
/// The weights blob is read **once** and kept in memory. An earlier version
/// re-read the whole file inside each accessor, which meant one full file read
/// per tensor: 79 reads of a 2.3 MB file to load one model, and far worse for
/// the 8.6 MB variant. On a microcontroller served by slow PSRAM that is not a
/// micro-optimisation, it is the difference between usable and not.
#[derive(Debug)]
pub struct Artifact {
    /// Parsed `meta.json`.
    pub meta: Meta,
    /// Directory the artifact was loaded from, for error messages.
    pub dir: PathBuf,
    blob: Vec<u8>,
    buffers: HashMap<String, RawBuffer>,
}

#[derive(Debug, Clone)]
/// A tensor's location and declared type inside the weights blob.
struct RawBuffer {
    /// safetensors dtype tag, e.g. `F32`.
    dtype: String,
    /// Shape declared in the weights header, cross-checked against `meta.json`.
    shape: Vec<usize>,
    /// Byte offset of the payload within the loaded blob.
    offset: usize,
    /// Payload length in bytes.
    len: usize,
}

/// Minimal safetensors header entry.
#[derive(Debug, Deserialize)]
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Artifact {
    /// Load an artifact directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref().to_path_buf();
        let meta_text =
            fs::read_to_string(dir.join("meta.json")).map_err(|e| format!("meta.json: {e}"))?;
        let meta: Meta = serde_json::from_str(&meta_text).map_err(|e| format!("meta.json: {e}"))?;
        if meta.artifact_version != 1 {
            return Err(format!(
                "unsupported artifact_version {}",
                meta.artifact_version
            ));
        }

        let blob = fs::read(dir.join("model.safetensors")).map_err(|e| format!("weights: {e}"))?;
        if blob.len() < 8 {
            return Err("weights: file too short for a safetensors header".into());
        }
        let header_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        if header_end > blob.len() {
            return Err("weights: header length exceeds file size".into());
        }
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&blob[8..header_end])
                .map_err(|e| format!("weights header: {e}"))?;

        let mut buffers = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry: StEntry =
                serde_json::from_value(value).map_err(|e| format!("weights[{name}]: {e}"))?;
            let [start, end] = entry.data_offsets;
            if end < start || header_end + end > blob.len() {
                return Err(format!(
                    "weights[{name}]: data_offsets {start}..{end} lie outside the buffer"
                ));
            }
            let len = end - start;
            // Cheap integrity check: the declared shape must account for exactly
            // the bytes present. Catches a truncated or mismatched file here
            // rather than as a wrong number at inference time.
            let elements: usize = entry.shape.iter().product();
            let width = match entry.dtype.as_str() {
                "F32" | "I32" => 4,
                "U8" | "I8" => 1,
                other => return Err(format!("weights[{name}]: unsupported dtype {other}")),
            };
            if elements * width != len {
                let shape = &entry.shape;
                return Err(format!(
                    "weights[{name}]: shape {shape:?} ({elements} elements) implies {} bytes, \
                     but {} are present",
                    elements * width,
                    len
                ));
            }
            buffers.insert(
                name,
                RawBuffer {
                    dtype: entry.dtype,
                    shape: entry.shape,
                    offset: header_end + start,
                    len,
                },
            );
        }

        Ok(Self {
            meta,
            dir,
            blob,
            buffers,
        })
    }

    /// Number of tensors in the artifact.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.meta.tensors.len()
    }

    /// Look up an f32 tensor by name. Quantized tensors are dequantized first.
    pub fn tensor(&self, name: &str) -> Result<Tensor, String> {
        let info = self
            .meta
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not in meta.json"))?;
        if info.quantized {
            crate::quant::dequantize(&self.quantized_weight(name)?)
        } else {
            // Cross-check the two independent descriptions of the shape: the
            // safetensors header and meta.json. They are written from different
            // sources, so a disagreement means the artifact is inconsistent.
            let buf_shape = &self.buffer(name)?.shape;
            if *buf_shape != info.shape {
                return Err(format!(
                    "{name}: shape disagreement - meta.json says {:?}, weights header says {:?}",
                    info.shape, buf_shape
                ));
            }
            let values = self.raw_f32(name)?;
            Ok(Tensor::new(values, info.shape.clone()))
        }
    }

    /// Assemble the codes, scales and zero points for a quantized weight.
    pub fn quantized_weight(&self, name: &str) -> Result<QuantizedWeight, String> {
        let info = self
            .meta
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not in meta.json"))?;
        if !info.quantized {
            return Err(format!("{name} is not quantized"));
        }
        let bits = info.bits.ok_or_else(|| format!("{name}: missing bits"))?;
        let granularity = info
            .granularity
            .as_deref()
            .and_then(Granularity::parse)
            .ok_or_else(|| format!("{name}: unknown granularity"))?;

        let packed = self.raw_bytes(&format!("{name}.q"))?.to_vec();
        let scale = self.raw_f32(&format!("{name}.scale"))?;
        let zero_point = self.raw_i32(&format!("{name}.zp"))?;

        Ok(QuantizedWeight {
            packed,
            scale,
            zero_point,
            shape: info.shape.clone(),
            bits,
            granularity,
            group_size: info.group_size.unwrap_or(1),
            axis: info.axis.unwrap_or(0),
        })
    }

    fn buffer(&self, name: &str) -> Result<&RawBuffer, String> {
        self.buffers
            .get(name)
            .ok_or_else(|| format!("weight tensor {name} not found in model.safetensors"))
    }

    /// Raw bytes of a tensor, sliced from the already-loaded blob.
    fn raw_bytes(&self, name: &str) -> Result<&[u8], String> {
        let buf = self.buffer(name)?;
        Ok(&self.blob[buf.offset..buf.offset + buf.len])
    }

    fn raw_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let buf = self.buffer(name)?;
        if buf.dtype != "F32" {
            return Err(format!("{name}: expected F32, got {}", buf.dtype));
        }
        Ok(self
            .raw_bytes(name)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn raw_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        Ok(self
            .raw_bytes(name)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Tensor shape from the index, for validation and reporting.
    #[must_use]
    pub fn shape_of(&self, name: &str) -> Option<&Vec<usize>> {
        self.meta.tensors.get(name).map(|i| &i.shape)
    }
}

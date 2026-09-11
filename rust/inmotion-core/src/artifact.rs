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
    pub quantized: bool,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub bits: Option<u32>,
    #[serde(default)]
    pub granularity: Option<String>,
    #[serde(default)]
    pub group_size: Option<usize>,
    #[serde(default)]
    pub axis: Option<usize>,
    #[serde(default)]
    pub scale_shape: Option<Vec<usize>>,
}

/// The `meta.json` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub artifact_version: u32,
    pub model: String,
    pub factory: String,
    pub rich_features: bool,
    pub seq_len: usize,
    pub in_channels: usize,
    pub num_classes: usize,
    pub eps: Eps,
    /// Architecture hyperparameters as exported (d_model, nhead, num_layers, ...).
    #[serde(default)]
    pub arch: HashMap<String, serde_json::Value>,
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(default)]
    pub quant: Option<serde_json::Value>,
}

/// Normalization epsilons, taken from the artifact rather than hardcoded.
#[derive(Debug, Clone, Deserialize)]
pub struct Eps {
    pub layernorm: f32,
    pub batchnorm: f32,
}

/// A loaded artifact.
#[derive(Debug)]
pub struct Artifact {
    pub meta: Meta,
    pub dir: PathBuf,
    buffers: HashMap<String, RawBuffer>,
}

#[derive(Debug, Clone)]
struct RawBuffer {
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
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
        let meta_text = fs::read_to_string(dir.join("meta.json"))
            .map_err(|e| format!("meta.json: {e}"))?;
        let meta: Meta = serde_json::from_str(&meta_text).map_err(|e| format!("meta.json: {e}"))?;
        if meta.artifact_version != 1 {
            return Err(format!("unsupported artifact_version {}", meta.artifact_version));
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
            serde_json::from_slice(&blob[8..header_end]).map_err(|e| format!("weights header: {e}"))?;

        let mut buffers = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry: StEntry =
                serde_json::from_value(value).map_err(|e| format!("weights[{name}]: {e}"))?;
            let [start, end] = entry.data_offsets;
            buffers.insert(
                name,
                RawBuffer {
                    dtype: entry.dtype,
                    shape: entry.shape,
                    offset: header_end + start,
                    len: end - start,
                },
            );
        }

        Ok(Self { meta, dir, buffers })
    }

    /// Number of tensors in the artifact.
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
            Ok(crate::quant::dequantize(&self.quantized_weight(name)?))
        } else {
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

        let packed = self.raw_bytes(&format!("{name}.q"))?;
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

    fn raw_bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        let buf = self.buffer(name)?;
        let blob = fs::read(self.dir.join("model.safetensors")).map_err(|e| e.to_string())?;
        Ok(blob[buf.offset..buf.offset + buf.len].to_vec())
    }

    fn raw_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let buf = self.buffer(name)?;
        if buf.dtype != "F32" {
            return Err(format!("{name}: expected F32, got {}", buf.dtype));
        }
        let mut out = Vec::with_capacity(buf.len / 4);
        for chunk in self.raw_slice(name)?.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }

    fn raw_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        let mut out = Vec::new();
        for chunk in self.raw_slice(name)?.chunks_exact(4) {
            out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }

    fn raw_slice(&self, name: &str) -> Result<Vec<u8>, String> {
        let buf = self.buffer(name)?;
        let blob = fs::read(self.dir.join("model.safetensors")).map_err(|e| e.to_string())?;
        Ok(blob[buf.offset..buf.offset + buf.len].to_vec())
    }

    /// Tensor shape from the index, for validation and reporting.
    pub fn shape_of(&self, name: &str) -> Option<&Vec<usize>> {
        self.meta.tensors.get(name).map(|i| &i.shape)
    }
}

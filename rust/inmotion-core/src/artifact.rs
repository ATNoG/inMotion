//! Reading the exported artifact: a safetensors weight bag plus `meta.json`.
//!
//! The safetensors reader is written here rather than pulled in as a
//! dependency. The format is a little-endian `u64` header length, a JSON header,
//! then the raw buffers, which is short enough to audit and keeps the dependency
//! set to `serde_json` alone.

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

impl Meta {
    /// Read an integer architecture field the exporter recorded.
    ///
    /// The runtime selects its code path from these fields, so a missing one is
    /// an error rather than a default: guessing produced a model that ran and
    /// returned confident wrong answers.
    pub fn usize_field(&self, key: &str) -> Result<usize, String> {
        let value = self
            .arch
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                format!("artifact does not record `{key}`; re-export it from the registry entry")
            })?;
        usize::try_from(value).map_err(|_| format!("`{key}` does not fit a usize"))
    }

    /// Read a boolean architecture field the exporter recorded.
    pub fn bool_field(&self, key: &str) -> Result<bool, String> {
        self.arch
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                format!("artifact does not record `{key}`; re-export it from the registry entry")
            })
    }

    /// Read a list-of-integers architecture field the exporter recorded.
    pub fn usize_list_field(&self, key: &str) -> Result<Vec<usize>, String> {
        let missing =
            || format!("artifact does not record `{key}`; re-export it from the registry entry");
        let items = self
            .arch
            .get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(missing)?;
        items
            .iter()
            .map(|item| {
                let value = item
                    .as_u64()
                    .ok_or_else(|| format!("`{key}` must be a list of integers, got {item}"))?;
                usize::try_from(value).map_err(|_| format!("`{key}` does not fit a usize"))
            })
            .collect()
    }

    /// Read a string architecture field the exporter recorded.
    pub fn str_field<'a>(&'a self, key: &str) -> Result<&'a str, String> {
        self.arch
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!("artifact does not record `{key}`; re-export it from the registry entry")
            })
    }
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
/// the 8.6 MB variant. On a router whose storage is far slower than its RAM that
/// is not a micro-optimisation, it is the difference between usable and not.
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

/// How many scale/zero-point pairs the declared granularity needs.
///
/// The same arithmetic `quant::dequantize` performs, applied up front so a
/// malformed artifact is rejected at load rather than indexing out of bounds in
/// the inner loop.
fn metadata_len(
    name: &str,
    shape: &[usize],
    granularity: Granularity,
    axis: usize,
    group_size: usize,
) -> Result<usize, String> {
    match granularity {
        Granularity::PerTensor => Ok(1),
        Granularity::PerChannel => shape.get(axis).copied().ok_or_else(|| {
            format!("{name}: per_channel axis {axis} is out of range for shape {shape:?}")
        }),
        Granularity::PerGroup => {
            if group_size == 0 {
                return Err(format!("{name}: per_group requires group_size > 0"));
            }
            let rows = shape
                .first()
                .copied()
                .ok_or_else(|| format!("{name}: per_group needs a leading axis"))?;
            if rows == 0 {
                return Err(format!("{name}: per_group has a zero-length leading axis"));
            }
            // Groups run along the flattened trailing axes, exactly as the
            // exporter's `_flatten_for_grouping` lays them out.
            let count: usize = shape.iter().product();
            let last = count / rows;
            rows.checked_mul(last.div_ceil(group_size))
                .ok_or_else(|| format!("{name}: per_group metadata count overflows"))
        }
    }
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
        let header_len = usize::try_from(u64::from_le_bytes(blob[..8].try_into().unwrap()))
            .map_err(|_| "weights: header length exceeds file size".to_string())?;
        let header_end = header_len
            .checked_add(8)
            .filter(|end| *end <= blob.len())
            .ok_or("weights: header length exceeds file size")?;
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
            let outside =
                || format!("weights[{name}]: data_offsets {start}..{end} lie outside the buffer");
            // `end` is attacker-controlled input; wrapping it would turn a bogus
            // header into a small in-bounds slice instead of a rejection.
            let buffer_end = header_end.checked_add(end).ok_or_else(outside)?;
            if end < start || buffer_end > blob.len() {
                return Err(outside());
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
        if scale.len() != zero_point.len() {
            return Err(format!(
                "{name}: {} scales but {} zero points",
                scale.len(),
                zero_point.len()
            ));
        }
        let group_size = info.group_size.unwrap_or(1);
        let axis = info.axis.unwrap_or(0);
        let expected_metadata = metadata_len(name, &info.shape, granularity, axis, group_size)?;
        if scale.len() != expected_metadata {
            return Err(format!(
                "{name}: {} scale/zero-point pairs, but a {granularity:?} weight of shape {:?} \
                 needs {expected_metadata}",
                scale.len(),
                info.shape
            ));
        }

        Ok(QuantizedWeight {
            packed,
            scale,
            zero_point,
            shape: info.shape.clone(),
            bits,
            granularity,
            group_size,
            axis,
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
        let buf = self.buffer(name)?;
        if buf.dtype != "I32" {
            return Err(format!("{name}: expected I32, got {}", buf.dtype));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_len_matches_the_granularity() {
        // per_tensor: one pair for the whole weight.
        assert_eq!(
            metadata_len("w", &[4, 8], Granularity::PerTensor, 0, 1).unwrap(),
            1
        );
        // per_channel: one pair per slice along the declared axis.
        assert_eq!(
            metadata_len("w", &[4, 8], Granularity::PerChannel, 1, 1).unwrap(),
            8
        );
        assert_eq!(
            metadata_len("w", &[4, 8], Granularity::PerChannel, 0, 1).unwrap(),
            4
        );
        // per_group: rows x groups over the flattened trailing axes, which for a
        // Conv1d (out, in, k) weight means grouping over in * k, not over k.
        assert_eq!(
            metadata_len("w", &[4, 8], Granularity::PerGroup, 0, 4).unwrap(),
            8
        );
        assert_eq!(
            metadata_len("w", &[256, 18, 2], Granularity::PerGroup, 0, 32).unwrap(),
            512
        );
    }

    #[test]
    fn metadata_len_rejects_declarations_it_cannot_honour() {
        // Axis out of range, zero group size, and no leading axis at all.
        assert!(metadata_len("w", &[4, 8], Granularity::PerChannel, 2, 1).is_err());
        assert!(metadata_len("w", &[4, 8], Granularity::PerGroup, 0, 0).is_err());
        assert!(metadata_len("w", &[], Granularity::PerGroup, 0, 4).is_err());
        assert!(metadata_len("w", &[0, 8], Granularity::PerGroup, 0, 4).is_err());
    }
}

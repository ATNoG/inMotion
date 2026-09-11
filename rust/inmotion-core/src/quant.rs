//! Unpacking and dequantization, mirroring `inmotion.quant.reference`.
//!
//! The exporter biases codes to unsigned (`u = q + 2**(bits-1)`) and folds that
//! bias into the stored zero point, so dequantization here is one formula for
//! every bit width: `(u - zero_point) * scale`. Keeping it uniform means the
//! device never branches on the bit width in its inner loop.

use crate::tensor::Tensor;

/// How many scale/zero-point entries the metadata carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    PerTensor,
    PerChannel,
    PerGroup,
}

impl Granularity {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "per_tensor" => Some(Self::PerTensor),
            "per_channel" => Some(Self::PerChannel),
            "per_group" => Some(Self::PerGroup),
            _ => None,
        }
    }
}

/// A quantized weight: packed codes plus the metadata to rebuild it.
#[derive(Clone, Debug)]
pub struct QuantizedWeight {
    pub packed: Vec<u8>,
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub shape: Vec<usize>,
    pub bits: u32,
    pub granularity: Granularity,
    pub group_size: usize,
    pub axis: usize,
}

/// Extract `count` codes of `bits` width from a packed LSB-first byte stream.
///
/// Mirrors `inmotion.quant.packing.unpack`: element `i` starts at bit
/// `i * bits`, so a code may straddle two bytes.
pub fn unpack(packed: &[u8], bits: u32, count: usize) -> Vec<u8> {
    debug_assert!((1..=8).contains(&bits));
    let mask: u32 = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bitpos = i * bits as usize;
        let byte = bitpos >> 3;
        let offset = bitpos & 7;
        let mut value = (packed[byte] as u32) >> offset;
        if offset + bits as usize > 8 {
            value |= (packed[byte + 1] as u32) << (8 - offset);
        }
        out.push((value & mask) as u8);
    }
    out
}

/// Rebuild the float weights, exactly as the Python reference does.
pub fn dequantize(qw: &QuantizedWeight) -> Tensor {
    let codes = unpack(&qw.packed, qw.bits, qw.shape.iter().product());
    let codes: Vec<f32> = codes.iter().map(|&u| u as f32).collect();

    match qw.granularity {
        Granularity::PerTensor => {
            let scale = qw.scale[0];
            let zp = qw.zero_point[0] as f32;
            Tensor::new(
                codes.iter().map(|&u| (u - zp) * scale).collect(),
                qw.shape.clone(),
            )
        }
        Granularity::PerChannel => {
            let axis = qw.axis;
            let mut strides = vec![1usize; qw.shape.len()];
            for i in (0..qw.shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * qw.shape[i + 1];
            }
            let stride = strides[axis];
            let dim = qw.shape[axis];
            let out: Vec<f32> = (0..codes.len())
                .map(|i| {
                    let index = (i / stride) % dim;
                    (codes[i] - qw.zero_point[index] as f32) * qw.scale[index]
                })
                .collect();
            Tensor::new(out, qw.shape.clone())
        }
        Granularity::PerGroup => {
            // Groups run along the last axis; >2-D weights were flattened by the
            // exporter, so the scale vector is (rows, n_groups).
            let last = *qw.shape.last().unwrap();
            let group = qw.group_size;
            let rows = codes.len() / last;
            let n_groups = qw.scale.len() / rows.max(1);
            let out: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    (0..last).map(move |c| (r, c))
                })
                .enumerate()
                .map(|(i, (r, c))| {
                    let g = c / group;
                    let _ = n_groups;
                    let meta = r * n_groups + g;
                    (codes[i] - qw.zero_point[meta] as f32) * qw.scale[meta]
                })
                .collect();
            Tensor::new(out, qw.shape.clone())
        }
    }
}

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
    /// One scale/zero-point pair for the whole tensor.
    PerTensor,
    /// One pair per slice along `axis`.
    PerChannel,
    /// One pair per contiguous run of `group_size` elements along the last axis.
    PerGroup,
}

impl Granularity {
    /// Parse the name the exporter writes, or `None` if it is unknown.
    #[must_use]
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
    /// Codes packed `bits` per element, LSB-first, biased to unsigned.
    pub packed: Vec<u8>,
    /// Dequantization scales, laid out according to `granularity`.
    pub scale: Vec<f32>,
    /// Zero points, paired with `scale`, already biased for unsigned codes.
    pub zero_point: Vec<i32>,
    /// Shape of the *original* weight, which the codes unpack back into.
    pub shape: Vec<usize>,
    /// Storage width of each code, 1..=8.
    pub bits: u32,
    /// How many codes share one scale/zero-point pair.
    pub granularity: Granularity,
    /// Elements per group when `granularity` is `PerGroup`.
    pub group_size: usize,
    /// Axis used when `granularity` is `PerChannel`.
    pub axis: usize,
}

/// Extract `count` codes of `bits` width from a packed LSB-first byte stream.
///
/// Mirrors `inmotion.quant.packing.unpack`: element `i` starts at bit
/// `i * bits`, so a code may straddle two bytes.
pub fn unpack(packed: &[u8], bits: u32, count: usize) -> Result<Vec<u8>, String> {
    if !(1..=8).contains(&bits) {
        return Err(format!("bits must be in 1..=8, got {bits}"));
    }
    let required = (count * bits as usize).div_ceil(8);
    if packed.len() < required {
        return Err(format!(
            "packed buffer too short: {count} codes of {bits} bits need {required} bytes, got {}",
            packed.len()
        ));
    }

    let mask: u32 = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bitpos = i * bits as usize;
        let byte = bitpos >> 3;
        let offset = bitpos & 7;
        let mut value = u32::from(packed[byte]) >> offset;
        if offset + bits as usize > 8 {
            // The code straddles two bytes; the high part comes from the next one.
            value |= u32::from(packed[byte + 1]) << (8 - offset);
        }
        out.push((value & mask) as u8);
    }
    Ok(out)
}

/// Rebuild the float weights, exactly as the Python reference does.
///
/// Every branch evaluates `(code - zero_point) * scale`, because the exporter
/// folds the unsigned bias into the zero point. Only the indexing differs.
pub fn dequantize(qw: &QuantizedWeight) -> Result<Tensor, String> {
    let count: usize = qw.shape.iter().product();
    let codes = unpack(&qw.packed, qw.bits, count)?;

    let out: Vec<f32> = match qw.granularity {
        Granularity::PerTensor => {
            let (scale, zp) = first_scale(qw)?;
            codes.iter().map(|&u| (f32::from(u) - zp) * scale).collect()
        }
        Granularity::PerChannel => {
            let axis = qw.axis;
            if axis >= qw.shape.len() {
                return Err(format!(
                    "per_channel axis {axis} out of range for {:?}",
                    qw.shape
                ));
            }
            let dim = qw.shape[axis];
            if qw.scale.len() != dim {
                return Err(format!(
                    "per_channel expects {dim} scales for axis {axis}, got {}",
                    qw.scale.len()
                ));
            }
            // Outer stride of `axis` in a row-major layout.
            let stride: usize = qw.shape[axis + 1..].iter().product();
            codes
                .iter()
                .enumerate()
                .map(|(i, &u)| {
                    let channel = (i / stride) % dim;
                    (f32::from(u) - qw.zero_point[channel] as f32) * qw.scale[channel]
                })
                .collect()
        }
        Granularity::PerGroup => {
            // Groups run along the last axis *after* the exporter flattened every
            // trailing axis into one, so a Conv1d (out, in, kernel) weight groups
            // over in*kernel, not over kernel. The artifact still declares the
            // original shape, so the row length is recovered from it rather than
            // from `shape.last()`. Metadata is laid out as (rows, n_groups) and
            // scale[i] pairs with the i-th group of the i / n_groups-th row.
            let group = qw.group_size;
            if group == 0 {
                return Err("per_group requires group_size > 0".into());
            }
            let rows = *qw.shape.first().ok_or("per_group needs a leading axis")?;
            if rows == 0 {
                return Err("per_group weight has a zero-length leading axis".into());
            }
            let last = count / rows;
            let n_groups = last.div_ceil(group);
            if qw.scale.len() != rows * n_groups {
                return Err(format!(
                    "per_group expects {} scales ({rows} rows x {n_groups} groups), got {}",
                    rows * n_groups,
                    qw.scale.len()
                ));
            }
            codes
                .iter()
                .enumerate()
                .map(|(i, &u)| {
                    let (row, col) = (i / last, i % last);
                    let meta = row * n_groups + col / group;
                    (f32::from(u) - qw.zero_point[meta] as f32) * qw.scale[meta]
                })
                .collect()
        }
    };

    Ok(Tensor::new(out, qw.shape.clone()))
}

/// The single scale/zero-point pair of a per-tensor weight.
fn first_scale(qw: &QuantizedWeight) -> Result<(f32, f32), String> {
    match (qw.scale.first(), qw.zero_point.first()) {
        (Some(&scale), Some(&zp)) => Ok((scale, zp as f32)),
        _ => Err("per_tensor weight has no scale/zero_point".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::Granularity::*;
    use super::*;

    /// A weight whose codes are `0..n` (zero-based) with metadata supplied
    /// directly, so the expected outputs below are `code * scale` with the
    /// zero point subtracted.
    fn weight(
        shape: Vec<usize>,
        bits: u32,
        granularity: Granularity,
        group_size: usize,
        axis: usize,
        scale: Vec<f32>,
        zero_point: Vec<i32>,
    ) -> QuantizedWeight {
        let n: usize = shape.iter().product();
        let codes: Vec<u8> = (0..n).map(|i| i as u8).collect();
        QuantizedWeight {
            packed: codes, // bits = 8 means the payload is the codes themselves
            scale,
            zero_point,
            shape,
            bits,
            granularity,
            group_size,
            axis,
        }
    }

    #[test]
    fn unpack_round_trips_for_every_bit_width() {
        // Codes biased to unsigned, exactly as `pack` writes them.
        for bits in 1..=8u32 {
            let limit = 1u16 << bits;
            let codes: Vec<u8> = (0..limit).map(|v| v as u8).collect();
            // Pack by hand: LSB-first, `bits` per element.
            let mut packed = vec![0u8; (codes.len() * bits as usize).div_ceil(8)];
            for (i, &value) in codes.iter().enumerate() {
                let bitpos = i * bits as usize;
                let byte = bitpos >> 3;
                let offset = bitpos & 7;
                packed[byte] |= (u16::from(value) << offset) as u8;
                if offset + bits as usize > 8 {
                    packed[byte + 1] |= (u16::from(value) >> (8 - offset)) as u8;
                }
            }
            let got = unpack(&packed, bits, codes.len()).expect("unpack");
            assert_eq!(got, codes, "bits={bits}");
        }
    }

    #[test]
    fn unpack_rejects_a_short_buffer() {
        // 16 four-bit codes need 8 bytes; give it 2.
        let err = unpack(&[0u8; 2], 4, 16).unwrap_err();
        assert!(err.contains("too short"), "{err}");
    }

    #[test]
    fn unpack_rejects_an_impossible_bit_width() {
        assert!(unpack(&[0u8; 4], 0, 1).is_err());
        assert!(unpack(&[0u8; 4], 9, 1).is_err());
    }

    #[test]
    fn per_tensor_applies_one_scale_and_zero_point() {
        let qw = weight(vec![3], 8, PerTensor, 1, 0, vec![2.0], vec![1]);
        let got = dequantize(&qw).unwrap();
        // codes 0,1,2 -> (0-1)*2, (1-1)*2, (2-1)*2
        assert_eq!(got.data, vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn per_channel_selects_by_the_outer_stride() {
        // shape (2,3), axis 0: elements 0..2 use channel 0 (scale 10), elements
        // 3..5 use channel 1 (scale 20). Codes are 0,1,2,3,4,5.
        let qw = weight(
            vec![2, 3],
            8,
            PerChannel,
            1,
            0,
            vec![10.0, 20.0],
            vec![0, 0],
        );
        let got = dequantize(&qw).unwrap();
        assert_eq!(got.data, vec![0.0, 10.0, 20.0, 60.0, 80.0, 100.0]);
    }

    #[test]
    fn per_channel_rejects_a_scale_count_that_does_not_match_the_axis() {
        let qw = weight(vec![2, 3], 8, PerChannel, 1, 0, vec![1.0], vec![0]);
        assert!(dequantize(&qw)
            .unwrap_err()
            .contains("per_channel expects 2 scales"));
    }

    #[test]
    fn per_group_walks_groups_along_the_last_axis() {
        // shape (2,4), group 2 -> 2 groups per row, metadata laid out row-major
        // as [row0g0, row0g1, row1g0, row1g1].
        let qw = weight(
            vec![2, 4],
            8,
            PerGroup,
            2,
            0,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 0, 0],
        );
        let got = dequantize(&qw).unwrap();
        // row 0 codes 0,1,2,3 -> groups [g0,g0,g1,g1] -> scales [1,1,2,2]
        // row 1 codes 4,5,6,7 -> groups [g0,g0,g1,g1] -> scales [3,3,4,4]
        assert_eq!(got.data, vec![0.0, 1.0, 4.0, 6.0, 12.0, 15.0, 24.0, 28.0]);
    }

    #[test]
    fn per_group_flattens_the_trailing_axes_of_a_conv_weight() {
        // A Conv1d (out=2, in=1, k=3) weight under per-group is grouped over the
        // flattened row of length 3, so the artifact's 3-D shape must not be read
        // as "groups run along the kernel axis".
        let qw = weight(
            vec![2, 1, 3],
            8,
            PerGroup,
            2,
            0,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0; 4],
        );
        let got = dequantize(&qw).unwrap();
        assert_eq!(got.shape, vec![2, 1, 3]);
        // Codes 0..5; row 0 = [0,1,2] -> groups [g0,g0,g1] -> scales [1,1,2];
        // row 1 = [3,4,5] -> groups [g0,g0,g1] -> scales [3,3,4].
        assert_eq!(got.data, vec![0.0, 1.0, 4.0, 9.0, 12.0, 20.0]);
    }

    #[test]
    fn per_group_rejects_a_scale_count_that_does_not_match_the_tiling() {
        let qw = weight(vec![2, 4], 8, PerGroup, 2, 0, vec![1.0, 2.0], vec![0, 0]);
        let err = dequantize(&qw).unwrap_err();
        assert!(err.contains("per_group expects 4 scales"), "{err}");
    }

    #[test]
    fn dequantize_output_shape_matches_the_declared_shape() {
        let qw = weight(vec![2, 3], 8, PerTensor, 1, 0, vec![1.0], vec![0]);
        let got = dequantize(&qw).unwrap();
        assert_eq!(got.shape, vec![2, 3]);
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn dequantize_reports_a_truncated_payload_instead_of_panicking() {
        let mut qw = weight(vec![2, 3], 8, PerTensor, 1, 0, vec![1.0], vec![0]);
        qw.packed.truncate(2);
        assert!(dequantize(&qw).is_err());
    }

    #[test]
    fn granularity_parses_the_names_the_exporter_writes() {
        assert_eq!(Granularity::parse("per_tensor"), Some(PerTensor));
        assert_eq!(Granularity::parse("per_channel"), Some(PerChannel));
        assert_eq!(Granularity::parse("per_group"), Some(PerGroup));
        assert_eq!(Granularity::parse("nonsense"), None);
    }

    #[test]
    fn zero_point_is_subtracted_before_scaling() {
        // Distinguishes (u - zp) * s from u * s - zp, which are not the same.
        let qw = weight(vec![2], 8, PerTensor, 1, 0, vec![3.0], vec![5]);
        let got = dequantize(&qw).unwrap();
        assert_eq!(got.data, vec![(0.0 - 5.0) * 3.0, (1.0 - 5.0) * 3.0]);
    }
}

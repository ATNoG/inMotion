//! The operators the JEPA encoder needs, nothing else.

use crate::tensor::Tensor;

/// `y = x @ w^T + b`, with `x: (T, in)`, `w: (out, in)`, `b: (out,)`.
#[must_use]
pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let (rows, in_features) = x.as_2d();
    let (out_features, w_in) = w.as_2d();
    assert_eq!(
        in_features, w_in,
        "linear: input width {in_features} != weight width {w_in}"
    );
    assert_eq!(out_features, b.len(), "linear: bias length != out_features");

    let mut out = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        let xr = &x.data[r * in_features..(r + 1) * in_features];
        for o in 0..out_features {
            let wr = &w.data[o * in_features..(o + 1) * in_features];
            let mut acc = b.data[o];
            for i in 0..in_features {
                acc += xr[i] * wr[i];
            }
            out[r * out_features + o] = acc;
        }
    }
    Tensor::new(out, vec![rows, out_features])
}

/// A weight matrix stored transposed, `(in_features, out_features)`.
///
/// [`linear`]'s inner loop is a dot product over the *input* axis. Each `acc +=`
/// depends on the previous one, so that is a serial `f32` chain the compiler
/// cannot vectorise, and on these models it is ~97 % of a forward pass. Sweeping
/// the *output* axis instead gives independent accumulators, which vectorise.
///
/// This is a layout change, not an approximation. The adds producing a given
/// output element still run in input-axis order, so a [`linear_packed`] result is
/// **bit-identical** to [`linear`]; `packed_matches_linear_bit_for_bit` pins it.
///
/// Measured against the alternative on the target hardware: at the router's
/// baseline `x86_64-unknown-linux-musl` target this kernel is 1.7x faster than
/// the `gemm` crate and 1.9x faster than `matrixmultiply`, both of which only
/// pull ahead when AVX2/FMA exist (the J3455 has neither).
#[derive(Debug, Clone)]
pub struct PackedWeight {
    /// Row-major `(in_features, out_features)`.
    data: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl PackedWeight {
    /// Transpose an `(out_features, in_features)` row-major weight.
    #[must_use]
    pub fn pack(w: &Tensor) -> Self {
        let (out_features, in_features) = w.as_2d();
        // The hand kernel sweeps the weight by input row, so it wants the
        // transposed (in, out) layout. The library kernels take the standard
        // row-major (out, in) one, so under those features the copy is not made.
        #[cfg(not(any(feature = "gemm", feature = "matrixmultiply")))]
        let data = {
            let mut packed = vec![0.0f32; in_features * out_features];
            for o in 0..out_features {
                for i in 0..in_features {
                    packed[i * out_features + o] = w.data[o * in_features + i];
                }
            }
            packed
        };
        #[cfg(any(feature = "gemm", feature = "matrixmultiply"))]
        let data = w.data[..in_features * out_features].to_vec();

        Self {
            data,
            in_features,
            out_features,
        }
    }
}

/// `y = x @ w + b`, with `w` already transposed by [`PackedWeight::pack`].
///
/// Bit-for-bit the same numbers as `linear(x, w_T, b)` where `w_T` is
/// `w` transposed back; see [`PackedWeight`].
#[must_use]
// `cast_possible_wrap`: these lengths are far below isize::MAX.
// `needless_return`: exactly one of several cfg'd alternatives is compiled, so
// the early ones must return rather than be the function's tail expression.
#[allow(clippy::cast_possible_wrap, clippy::needless_return)]
pub fn linear_packed(x: &Tensor, w: &PackedWeight, b: &Tensor) -> Tensor {
    let (rows, in_features) = x.as_2d();
    let out_features = w.out_features;
    assert_eq!(
        in_features, w.in_features,
        "linear_packed: input width {in_features} != weight width {}",
        w.in_features
    );
    assert_eq!(
        out_features,
        b.len(),
        "linear_packed: bias length != out_features"
    );

    // Every accumulator starts at the bias, exactly as the scalar kernel does, so
    // the two agree bit for bit rather than merely to within an epsilon.
    let mut acc = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        acc[r * out_features..(r + 1) * out_features].copy_from_slice(&b.data);
    }
    #[cfg(feature = "matrixmultiply")]
    {
        // SAFETY: every pointer is derived from a live slice, the strides walk
        // exactly those slices, and `matrixmultiply` only reads A and B and
        // writes C. A is (out, in) row-major; B is the input as (in, rows)
        // column-major, which is the (rows, in) buffer read transposed, so no
        // copy is made; C is (out, rows) column-major, i.e. `acc` read
        // row-major. beta = 1.0 because the bias is already in the accumulator,
        // matching the hand kernel's starting point.
        #[allow(unsafe_code)]
        unsafe {
            matrixmultiply::sgemm(
                out_features,
                in_features,
                rows,
                1.0,
                w.data.as_ptr(),
                in_features as isize,
                1,
                x.data.as_ptr(),
                1,
                in_features as isize,
                1.0,
                acc.as_mut_ptr(),
                1,
                out_features as isize,
            );
        }
        return Tensor::new(acc, vec![rows, out_features]);
    }

    #[cfg(feature = "rayon")]
    {
        // Parallel over output rows, which is bit-exact: each row still
        // accumulates over the input axis in the same order, so only
        // independent work is spread out. A batch-1 inference has few rows, so
        // this is expected to help throughput rather than single-call latency.
        use rayon::prelude::*;

        let x_data = &x.data;
        let w_data = &w.data;
        acc.par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(r, row)| {
                for i in 0..in_features {
                    let xi = x_data[r * in_features + i];
                    let wr = &w_data[i * out_features..(i + 1) * out_features];
                    for o in 0..out_features {
                        row[o] += xi * wr[o];
                    }
                }
            });
        return Tensor::new(acc, vec![rows, out_features]);
    }

    // The default backend. The whole tail is inside the cfg so that, when one of
    // the blocks above has already returned, there is no unreachable expression
    // left for the compiler to warn about.
    #[cfg(not(any(feature = "matrixmultiply", feature = "rayon")))]
    {
        // Rank-1 sweeps: one row of the weight feeds every output row, so the
        // weight is read once per call and the whole accumulator set stays in L1.
        for i in 0..in_features {
            let wr = &w.data[i * out_features..(i + 1) * out_features];
            for r in 0..rows {
                let xi = x.data[r * in_features + i];
                let ar = &mut acc[r * out_features..(r + 1) * out_features];
                for o in 0..out_features {
                    ar[o] += xi * wr[o];
                }
            }
        }
        Tensor::new(acc, vec![rows, out_features])
    }
}

/// Exact GELU: `0.5 x (1 + erf(x / sqrt(2)))`.
///
/// `PyTorch`'s `nn.GELU()` default is the erf form, not the `tanh` approximation,
/// so the same form is required here for parity.
#[must_use]
pub fn gelu(x: &Tensor) -> Tensor {
    let inv_sqrt2 = 1.0f32 / libm::sqrtf(2.0);
    Tensor::new(
        x.data
            .iter()
            .map(|&v| 0.5 * v * (1.0 + libm::erff(v * inv_sqrt2)))
            .collect(),
        x.shape.clone(),
    )
}

/// `LayerNorm` over the last axis, with `PyTorch`'s biased variance.
#[must_use]
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let width = x.last_dim();
    assert_eq!(width, weight.len(), "layer_norm: width != weight length");
    let rows = x.rows();
    let mut out = vec![0.0f32; x.len()];

    for r in 0..rows {
        let row = &x.data[r * width..(r + 1) * width];
        let mean = row.iter().sum::<f32>() / width as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let inv = 1.0 / libm::sqrtf(var + eps);
        for i in 0..width {
            out[r * width + i] = (row[i] - mean) * inv * weight.data[i] + bias.data[i];
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// Fold `BatchNorm1d` in eval mode into a per-channel `(scale, shift)`.
///
/// `BatchNorm1d` normalises with `(x - mean) / sqrt(var + eps)` and then applies
/// `weight`/`bias`, which is an affine map. Computing its two coefficients once
/// at load keeps the running statistics out of the inner loop, so inference is
/// one multiply-add per element.
#[must_use]
pub fn fold_batch_norm(
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let channels = weight.len();
    assert_eq!(
        bias.len(),
        channels,
        "batch_norm: bias length != weight length"
    );
    assert_eq!(
        running_mean.len(),
        channels,
        "batch_norm: running_mean length != weight length"
    );
    assert_eq!(
        running_var.len(),
        channels,
        "batch_norm: running_var length != weight length"
    );

    // Multiply by the reciprocal rather than dividing: the two round differently
    // and the multiplication is the form the pre-fold code used, so lejepa's
    // verified parity numbers stay bit-identical.
    let scale: Vec<f32> = weight
        .data
        .iter()
        .zip(&running_var.data)
        .map(|(w, var)| w * (1.0 / libm::sqrtf(var + eps)))
        .collect();
    let shift: Vec<f32> = bias
        .data
        .iter()
        .zip(running_mean.data.iter().zip(&scale))
        .map(|(b, (mean, s))| b - mean * s)
        .collect();
    (scale, shift)
}

/// Apply a folded per-channel affine to a `(channels, length)` tensor.
#[must_use]
pub fn channel_affine(x: &Tensor, scale: &[f32], shift: &[f32]) -> Tensor {
    let channels = x.shape[0];
    let length = x.data.len() / channels;
    assert_eq!(
        scale.len(),
        channels,
        "channel_affine: scale length != channels"
    );
    assert_eq!(
        shift.len(),
        channels,
        "channel_affine: shift length != channels"
    );

    let mut out = vec![0.0f32; x.len()];
    for c in 0..channels {
        for t in 0..length {
            out[c * length + t] = x.data[c * length + t] * scale[c] + shift[c];
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// `BatchNorm1d` in eval mode: fold the running statistics, then apply.
///
/// `x` is `(channels, length)` in the layout the encoders use after their
/// transpose. Callers that can afford to keep the coefficients (the CNN does)
/// should use [`fold_batch_norm`] + [`channel_affine`] directly.
#[must_use]
pub fn batch_norm_eval(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f32,
) -> Tensor {
    let (scale, shift) = fold_batch_norm(weight, bias, running_mean, running_var, eps);
    channel_affine(x, &scale, &shift)
}

/// Numerically stable softmax over the last axis.
pub fn softmax_last(x: &Tensor) -> Tensor {
    let width = x.last_dim();
    let rows = x.rows();
    let mut out = vec![0.0f32; x.len()];

    for r in 0..rows {
        let row = &x.data[r * width..(r + 1) * width];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..width {
            let e = libm::expf(row[i] - max);
            out[r * width + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for i in 0..width {
            out[r * width + i] *= inv;
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// Add two equal-shaped tensors elementwise.
#[must_use]
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.len(),
        b.len(),
        "add: shape mismatch {:?} vs {:?}",
        a.shape,
        b.shape
    );
    Tensor::new(
        a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
        a.shape.clone(),
    )
}

/// Transpose a 2-D tensor. A reshape is not a transpose: reinterpreting
/// `(T, C)` memory as `(C, T)` silently reorders every element.
#[must_use]
pub fn transpose_2d(x: &Tensor) -> Tensor {
    let (rows, cols) = x.as_2d();
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x.data[r * cols + c];
        }
    }
    Tensor::new(out, vec![cols, rows])
}

/// Softmax over the *first* axis of a `(N, 1)` tensor.
///
/// The attention-pooling head projects to a single score per timestep and then
/// applies `softmax(dim=1)` over the timestep axis, so the reduction runs along
/// T, not along the trailing singleton. Applying the usual last-axis softmax
/// there would return all ones and produce a plain mean.
pub fn softmax_first_axis(x: &Tensor) -> Tensor {
    let (rows, cols) = x.as_2d();
    assert_eq!(cols, 1, "softmax_first_axis expects a single column");
    let max = x.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(rows);
    let mut sum = 0.0f32;
    for value in &x.data {
        let e = libm::expf(value - max);
        exps.push(e);
        sum += e;
    }
    let inv = 1.0 / sum;
    Tensor::new(exps.into_iter().map(|e| e * inv).collect(), vec![rows, 1])
}

/// Multi-head self-attention for a single sequence, `x: (T, d_model)`.
///
/// All four projections are `(d_model, d_model)`, already transposed into the
/// layout [`linear_packed`] wants. They are separate tensors because the exporter
/// splits `PyTorch`'s packed `in_proj_weight`; the reader should not have to know
/// that convention.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn self_attention(
    x: &Tensor,
    q_w: &PackedWeight,
    q_b: &Tensor,
    k_w: &PackedWeight,
    k_b: &Tensor,
    v_w: &PackedWeight,
    v_b: &Tensor,
    out_w: &PackedWeight,
    out_b: &Tensor,
    n_heads: usize,
) -> Tensor {
    let (t, d_model) = x.as_2d();
    assert_eq!(
        d_model % n_heads,
        0,
        "d_model {d_model} not divisible by {n_heads} heads"
    );
    let head_dim = d_model / n_heads;
    let scale = 1.0 / libm::sqrtf(head_dim as f32);

    let q = linear_packed(x, q_w, q_b);
    let k = linear_packed(x, k_w, k_b);
    let v = linear_packed(x, v_w, v_b);

    let mut context = vec![0.0f32; t * d_model];
    for (name, w) in [("q", q_w), ("k", k_w), ("v", v_w), ("out", out_w)] {
        assert_eq!(
            w.out_features, d_model,
            "self_attention: {name} projection is not (d_model, d_model)"
        );
    }
    for h in 0..n_heads {
        let offset = h * head_dim;

        // `scores[i][j] = q_i . k_j * scale`. Written with the reduction inside
        // the `j` loop this is one serial `f32` chain per (i, j) and does not
        // vectorise; at T = 182 that alone cost 15 ms per layer, 11x all four
        // projections together. Sweeping `j` contiguously instead needs `k`'s
        // head slice laid out `(head_dim, t)`, because it is strided in `j`.
        //
        // The sum for a given (i, j) still runs over `c` in order, so this
        // produces the same bits as the triple loop it replaces.
        let mut kh = vec![0.0f32; head_dim * t];
        for c in 0..head_dim {
            for j in 0..t {
                kh[c * t + j] = k.data[j * d_model + offset + c];
            }
        }
        let mut scores = vec![0.0f32; t * t];
        for i in 0..t {
            let qrow = &q.data[i * d_model + offset..i * d_model + offset + head_dim];
            let srow = &mut scores[i * t..(i + 1) * t];
            for (c, &qc) in qrow.iter().enumerate() {
                let krow = &kh[c * t..(c + 1) * t];
                for j in 0..t {
                    srow[j] += qc * krow[j];
                }
            }
            for score in srow.iter_mut() {
                *score *= scale;
            }
        }
        let probs = softmax_last(&Tensor::new(scores, vec![t, t]));

        // `context[i][c] = sum_j probs[i][j] * v[j][c]`, again with the
        // reduction inside the `c` loop in the original. Each row of `v` is
        // contiguous in `c`, so holding one accumulator row and walking `j`
        // outward vectorises while leaving the `j`-order of the adds intact.
        for i in 0..t {
            let prow = &probs.data[i * t..(i + 1) * t];
            let crow = &mut context[i * d_model + offset..i * d_model + offset + head_dim];
            for (j, &p) in prow.iter().enumerate() {
                let vrow = &v.data[j * d_model + offset..j * d_model + offset + head_dim];
                for c in 0..head_dim {
                    crow[c] += p * vrow[c];
                }
            }
        }
    }

    linear_packed(&Tensor::new(context, vec![t, d_model]), out_w, out_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec())
    }

    #[test]
    fn linear_matches_hand_computation() {
        // [[1,2],[3,4]] @ I + 0 with the identity weight
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = t(&[0.0, 0.0], &[2]);
        let y = linear(&x, &w, &b);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 4.0]);

        // `linear` computes x @ w^T, so out[o] = x . w[o] (row o of the weight).
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]); // rows [1,2] and [3,4]
        let b = t(&[0.0, 0.0], &[2]);
        let y = linear(&x, &w, &b);
        assert_eq!(y.data, vec![1.0 * 1.0 + 2.0 * 2.0, 1.0 * 3.0 + 2.0 * 4.0]);
        assert_eq!(y.data, vec![5.0, 11.0]);

        // Bias is added once per output unit.
        let b = t(&[10.0, -10.0], &[2]);
        let y = linear(&x, &w, &b);
        assert_eq!(y.data, vec![15.0, 1.0]);
    }

    #[test]
    fn linear_accepts_a_batched_row_count() {
        let x = t(&[1.0, 1.0, 2.0, 2.0], &[2, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = t(&[0.0, 0.0], &[2]);
        assert_eq!(linear(&x, &w, &b).data, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn gelu_matches_the_erf_definition() {
        // Reference values for exact (erf) GELU.
        assert!((gelu(&t(&[0.0], &[1])).data[0]).abs() < 1e-7);
        // 0.841345 is the exact erf-form value. The commonly quoted 0.841192 is
        // the tanh approximation, which PyTorch does not use by default.
        assert!((gelu(&t(&[1.0], &[1])).data[0] - 0.841_345).abs() < 1e-5);
        assert!((gelu(&t(&[-1.0], &[1])).data[0] + 0.158_655).abs() < 1e-5);
        // Large positive input approaches the identity.
        assert!((gelu(&t(&[6.0], &[1])).data[0] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_standardises_each_row() {
        // Two values per row: mean 1.5, variance 0.25, standard deviation 0.5,
        // so the standardised values are exactly -1 and +1.
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = t(&[1.0, 1.0], &[2]);
        let b = t(&[0.0, 0.0], &[2]);
        let y = layer_norm(&x, &w, &b, 0.0);
        assert!((y.data[0] + 1.0).abs() < 1e-6, "{:?}", y.data);
        assert!((y.data[1] - 1.0).abs() < 1e-6);
        assert!((y.data[2] + 1.0).abs() < 1e-6);
        assert!((y.data[3] - 1.0).abs() < 1e-6);
        // Every row is zero-mean by construction.
        assert!((y.data[0] + y.data[1]).abs() < 1e-6);
        assert!((y.data[2] + y.data[3]).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_applies_affine_parameters() {
        let x = t(&[1.0, 3.0], &[1, 2]);
        let w = t(&[2.0, 2.0], &[2]);
        let b = t(&[1.0, 1.0], &[2]);
        let y = layer_norm(&x, &w, &b, 0.0);
        // mean 2, var 1 -> z = [-1, 1] -> 2z + 1 = [-1, 3]
        assert!((y.data[0] + 1.0).abs() < 1e-6);
        assert!((y.data[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_a_distribution_and_is_shift_invariant() {
        let y = softmax_last(&t(&[1.0, 2.0, 3.0], &[1, 3]));
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(y.data.iter().all(|v| *v > 0.0));

        // Adding a constant must not change the result.
        let z = softmax_last(&t(&[101.0, 102.0, 103.0], &[1, 3]));
        assert!(y.max_abs_diff(&z) < 1e-6);
    }

    #[test]
    fn softmax_survives_large_magnitudes() {
        // Naive exponentiation would overflow here.
        let y = softmax_last(&t(&[1000.0, 1001.0, 1002.0], &[1, 3]));
        assert!(y.data.iter().all(|v| v.is_finite()));
        assert!((y.data.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_first_axis_normalises_over_timesteps() {
        // The pooling head produces one score per timestep and reduces over T.
        let y = softmax_first_axis(&t(&[0.0, 0.0, 0.0, 0.0], &[4, 1]));
        assert_eq!(y.data, vec![0.25; 4]);

        let y = softmax_first_axis(&t(&[0.0, 0.0], &[2, 1]));
        assert_eq!(y.data, vec![0.5, 0.5]);
    }

    #[test]
    fn transpose_actually_reorders_memory() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = transpose_2d(&x);
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Transposing twice restores the original.
        assert_eq!(transpose_2d(&y).data, x.data);
    }

    #[test]
    fn batch_norm_eval_is_an_affine_map() {
        // weight 1, bias 0, mean 0, var 1, eps 0 -> identity.
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let ones = t(&[1.0, 1.0], &[2]);
        let zeros = t(&[0.0, 0.0], &[2]);
        let y = batch_norm_eval(&x, &ones, &zeros, &zeros, &ones, 0.0);
        assert!(x.max_abs_diff(&y) < 1e-6);

        // Shifting the mean subtracts it.
        let mean = t(&[1.0, 1.0], &[2]);
        let y = batch_norm_eval(&x, &ones, &zeros, &mean, &ones, 0.0);
        assert!((y.data[0] - 0.0).abs() < 1e-6);
        assert!((y.data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn conv1d_same_zero_pads_to_keep_the_length() {
        // [1,2,3,4] with an all-ones kernel 3 and padding 1: every output is the
        // sum of three consecutive taps, with zeros outside the signal.
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let w = t(&[1.0, 1.0, 1.0], &[1, 1, 3]);
        let y = conv1d_same(&x, &w, &t(&[0.0], &[1]));
        assert_eq!(y.shape, vec![1, 4]);
        assert_eq!(y.data, vec![3.0, 6.0, 9.0, 7.0]);
        // Bias is added at every output position.
        assert_eq!(
            conv1d_same(&x, &w, &t(&[10.0], &[1])).data,
            vec![13.0, 16.0, 19.0, 17.0]
        );
    }

    #[test]
    fn conv1d_same_reads_each_input_channel() {
        // ch0 = [1,2,3], ch1 = [10,20,30]; the kernel weights channel 1 only.
        let x = t(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3]);
        let w = t(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[1, 2, 3]);
        let y = conv1d_same(&x, &w, &t(&[0.0], &[1]));
        // Three consecutive taps of channel 1, zero-padded at both ends.
        assert_eq!(y.shape, vec![1, 3]);
        assert_eq!(y.data, vec![30.0, 60.0, 50.0]);
    }

    #[test]
    fn conv1d_same_skips_taps_that_fall_entirely_in_the_padding() {
        // A signal shorter than the kernel: length 1, kernel 3, padding 1. Only
        // the centre tap (k = 1) reads the signal; the taps at k = 0 and k = 2
        // fall outside it and must contribute nothing.
        let x = t(&[5.0], &[1, 1]);
        let w = t(&[1.0, 2.0, 4.0], &[1, 1, 3]);
        let y = conv1d_same(&x, &w, &t(&[0.0], &[1]));
        assert_eq!(y.shape, vec![1, 1]);
        assert_eq!(y.data, vec![10.0]);
    }

    #[test]
    fn concat_channels_joins_along_the_first_axis() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0], &[1, 2]);
        let y = concat_channels(&[a, b]);
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn fold_batch_norm_matches_the_definition() {
        // weight 2, bias 1, mean 3, var 4, eps 0 -> scale 2/2 = 1, shift 1-3 = -2.
        let w = t(&[2.0], &[1]);
        let b = t(&[1.0], &[1]);
        let mean = t(&[3.0], &[1]);
        let var = t(&[4.0], &[1]);
        let (scale, shift) = fold_batch_norm(&w, &b, &mean, &var, 0.0);
        assert!((scale[0] - 1.0).abs() < 1e-6, "{scale:?}");
        assert!((shift[0] + 2.0).abs() < 1e-6, "{shift:?}");

        // The folded pair equals the per-element formula for arbitrary input.
        let x = t(&[1.0, 5.0, -3.0], &[1, 3]);
        let folded = channel_affine(&x, &scale, &shift);
        for (i, &v) in x.data.iter().enumerate() {
            let expected = (v - 3.0) / (4.0f32).sqrt() * 2.0 + 1.0;
            assert!((folded.data[i] - expected).abs() < 1e-5);
        }
    }

    /// Identity projections, no bias, one head.
    fn identity_attention(x: &Tensor, d: usize) -> Tensor {
        let eye = PackedWeight::pack(&t(&[1.0, 0.0, 0.0, 1.0], &[d, d]));
        let zeros = t(&[0.0, 0.0], &[d]);
        self_attention(x, &eye, &zeros, &eye, &zeros, &eye, &zeros, &eye, &zeros, 1)
    }

    #[test]
    fn packed_matches_linear_bit_for_bit() {
        // Every shape the models use, plus a non-square one and a single output
        // unit (`attn.weight` is `(1, d_model)`). The two kernels must agree
        // exactly, not approximately: the whole point of the layout change is
        // that it is free.
        for (rows, k, n) in [
            (10usize, 128usize, 128usize),
            (10, 128, 256),
            (10, 256, 128),
            (5, 256, 256),
            (10, 4, 128),
            (10, 192, 128),
            (10, 128, 1),
            (1, 128, 4),
        ] {
            let x = t(
                &(0..rows * k)
                    .map(|i| (i * 37 % 41) as f32 * 0.1 - 2.0)
                    .collect::<Vec<_>>(),
                &[rows, k],
            );
            let w = t(
                &(0..n * k)
                    .map(|i| (i * 17 % 23) as f32 * 0.01 - 0.1)
                    .collect::<Vec<_>>(),
                &[n, k],
            );
            let b = t(
                &(0..n)
                    .map(|i| (i * 7 % 11) as f32 * 0.01)
                    .collect::<Vec<_>>(),
                &[n],
            );
            let expected = linear(&x, &w, &b);
            let got = linear_packed(&x, &PackedWeight::pack(&w), &b);
            assert_eq!(got.shape, expected.shape, "({rows},{k})->{n}");
            assert_eq!(
                got.data, expected.data,
                "({rows},{k})->{n} is not bit-exact"
            );
        }
    }

    #[test]
    fn packed_linear_adds_the_bias_before_the_products() {
        // The bias is the *initial* value of each accumulator, matching `linear`.
        // Seeding from zero and adding the bias last is a different rounding and
        // would break the bit-exactness the test above pins.
        let x = t(&[2.0], &[1, 1]);
        let w = t(&[3.0], &[1, 1]);
        let b = t(&[0.5], &[1]);
        let got = linear_packed(&x, &PackedWeight::pack(&w), &b);
        assert_eq!(got.data, vec![6.5]);
    }

    /// The plain triple loop `self_attention` used before it was rewritten, kept
    /// as the oracle for `attention_is_bit_exact_after_the_rewrite`.
    fn scalar_attention(x: &Tensor, w: &[(&Tensor, &Tensor); 4], n_heads: usize) -> Tensor {
        let (t, d_model) = x.as_2d();
        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let project = |i: usize| linear(x, w[i].0, w[i].1);
        let (q, k, v) = (project(0), project(1), project(2));

        let mut context = vec![0.0f32; t * d_model];
        for h in 0..n_heads {
            let offset = h * head_dim;
            let mut scores = vec![0.0f32; t * t];
            for i in 0..t {
                for j in 0..t {
                    let mut acc = 0.0f32;
                    for c in 0..head_dim {
                        acc += q.data[i * d_model + offset + c] * k.data[j * d_model + offset + c];
                    }
                    scores[i * t + j] = acc * scale;
                }
            }
            let probs = softmax_last(&Tensor::new(scores, vec![t, t]));
            for i in 0..t {
                for c in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..t {
                        acc += probs.data[i * t + j] * v.data[j * d_model + offset + c];
                    }
                    context[i * d_model + offset + c] = acc;
                }
            }
        }
        linear(&Tensor::new(context, vec![t, d_model]), w[3].0, w[3].1)
    }

    #[test]
    fn attention_is_bit_exact_after_the_rewrite() {
        // The rewrite reorders only the *loops*, never the adds that make up a
        // given output element, so it must not move a single bit. Several token
        // counts and head widths, because the old code's cost grew with T while
        // the new code's does not.
        for (t, d_model, n_heads) in [(1usize, 4usize, 1usize), (3, 4, 2), (10, 8, 4), (7, 6, 3)] {
            let fill = |n: usize, m: usize, seed: usize| {
                Tensor::new(
                    (0..n * m)
                        .map(|i| ((i * 17 + seed * 5) % 29) as f32 * 0.05 - 0.7)
                        .collect(),
                    vec![n, m],
                )
            };
            let x = fill(t, d_model, 1);
            let weights: Vec<(Tensor, Tensor)> = (0..4)
                .map(|s| (fill(d_model, d_model, s + 2), fill(d_model, 1, s + 9)))
                .collect();
            let packed: Vec<PackedWeight> =
                weights.iter().map(|(w, _)| PackedWeight::pack(w)).collect();

            let expected = scalar_attention(
                &x,
                &[
                    (&weights[0].0, &weights[0].1),
                    (&weights[1].0, &weights[1].1),
                    (&weights[2].0, &weights[2].1),
                    (&weights[3].0, &weights[3].1),
                ],
                n_heads,
            );
            let got = self_attention(
                &x,
                &packed[0],
                &weights[0].1,
                &packed[1],
                &weights[1].1,
                &packed[2],
                &weights[2].1,
                &packed[3],
                &weights[3].1,
                n_heads,
            );
            assert_eq!(
                got.data, expected.data,
                "t={t} d_model={d_model} n_heads={n_heads} is not bit-exact"
            );
        }
    }

    #[test]
    fn self_attention_with_a_single_token_returns_that_token() {
        // T = 1: softmax over one element is 1, so the output is V, which is x.
        let x = t(&[3.0, -2.0], &[1, 2]);
        let y = identity_attention(&x, 2);
        assert!(x.max_abs_diff(&y) < 1e-6, "{:?}", y.data);
    }

    #[test]
    fn self_attention_with_identical_tokens_returns_that_token() {
        // Equal queries and keys make every score equal, so attention is uniform
        // and the output is the mean of V, which is the repeated row itself.
        let x = t(&[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], &[3, 2]);
        let y = identity_attention(&x, 2);
        for row in 0..3 {
            assert!((y.data[row * 2] - 1.0).abs() < 1e-5, "{:?}", y.data);
            assert!((y.data[row * 2 + 1] - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn self_attention_output_is_a_convex_combination_of_values() {
        // With identity V projections the output must lie within the range of the
        // input rows, because attention weights are a distribution over them.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let y = identity_attention(&x, 2);
        for value in &y.data {
            assert!(*value >= 0.9 && *value <= 6.1, "out of range: {:?}", y.data);
        }
    }
}

/// 1-D convolution over a `(channels_in, length)` signal.
///
/// Only what the patch tokenizer needs: no padding, no dilation, explicit
/// stride. Output is `(channels_out, out_len)` with
/// `out_len = (length - kernel_size) / stride + 1`.
#[must_use]
pub fn conv1d(x: &Tensor, weight: &Tensor, bias: &Tensor, stride: usize) -> Tensor {
    assert_eq!(x.shape.len(), 2, "conv1d expects (channels, length)");
    assert_eq!(
        weight.shape.len(),
        3,
        "conv1d weight must be (out, in, kernel)"
    );
    let (channels_in, length) = (x.shape[0], x.shape[1]);
    let (channels_out, w_in, kernel) = (weight.shape[0], weight.shape[1], weight.shape[2]);
    assert_eq!(
        channels_in, w_in,
        "conv1d: input channels {channels_in} != weight {w_in}"
    );
    assert_eq!(
        bias.len(),
        channels_out,
        "conv1d: bias length != out channels"
    );
    assert!(length >= kernel, "conv1d: signal shorter than the kernel");
    let out_len = (length - kernel) / stride + 1;

    let mut out = vec![0.0f32; channels_out * out_len];
    for co in 0..channels_out {
        for t in 0..out_len {
            let mut acc = bias.data[co];
            for ci in 0..channels_in {
                for k in 0..kernel {
                    let value = x.data[ci * length + t * stride + k];
                    let w = weight.data[(co * channels_in + ci) * kernel + k];
                    acc += value * w;
                }
            }
            out[co * out_len + t] = acc;
        }
    }
    Tensor::new(out, vec![channels_out, out_len])
}

/// 1-D convolution with `padding = kernel_size / 2`, so the output keeps the
/// input length.
///
/// This is `nn.Conv1d(..., padding=k//2)` as the `SIGReg` CNN uses it. Padding
/// is zero, which the inner loop handles by skipping out-of-range taps rather
/// than materialising a padded copy of the signal.
#[must_use]
pub fn conv1d_same(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "conv1d_same expects (channels, length)");
    assert_eq!(
        weight.shape.len(),
        3,
        "conv1d_same weight must be (out, in, kernel)"
    );
    let (channels_in, length) = (x.shape[0], x.shape[1]);
    let (channels_out, w_in, kernel) = (weight.shape[0], weight.shape[1], weight.shape[2]);
    assert_eq!(
        channels_in, w_in,
        "conv1d_same: input channels {channels_in} != weight {w_in}"
    );
    assert_eq!(
        bias.len(),
        channels_out,
        "conv1d_same: bias length != out channels"
    );
    let pad = kernel / 2;

    // One output element reduces over `ci` then `k`. Written as the innermost
    // loop that is a separate serial `f32` chain per element and does not
    // vectorise; it was ~100 % of this family's inference time.
    //
    // Flattening `(ci, k)` into one axis `j` and sweeping the *output channel*
    // axis innermost fixes that without reordering any adds: for a given
    // `(co, t)` the accumulation still walks `j = 0, 1, 2, ...`, which is
    // `ci`-major then `k`-minor, exactly as before. Only the loops move.
    let taps = channels_in * kernel;
    let mut by_tap = vec![0.0f32; taps * channels_out];
    for ci in 0..channels_in {
        for k in 0..kernel {
            let dst = &mut by_tap[(ci * kernel + k) * channels_out..][..channels_out];
            for (co, slot) in dst.iter_mut().enumerate() {
                *slot = weight.data[(co * channels_in + ci) * kernel + k];
            }
        }
    }

    // Accumulate transposed, `(length, channels_out)`, so the inner loop is
    // contiguous; the caller's layout is restored on the way out.
    let mut acc = vec![0.0f32; length * channels_out];
    for t in 0..length {
        acc[t * channels_out..(t + 1) * channels_out].copy_from_slice(&bias.data);
    }
    for j in 0..taps {
        let (ci, k) = (j / kernel, j % kernel);
        // Tap `k` reads source position `t + k - pad`; the positions before the
        // signal or past its end are the zero padding, so the valid `t` form a
        // single interval and the inner loop needs no branch.
        let t0 = pad.saturating_sub(k);
        let t1 = (length + pad).saturating_sub(k).min(length);
        if t0 >= t1 {
            continue;
        }
        let src0 = t0 + k - pad;
        let wrow = &by_tap[j * channels_out..(j + 1) * channels_out];
        for t in t0..t1 {
            let value = x.data[ci * length + src0 + (t - t0)];
            let arow = &mut acc[t * channels_out..(t + 1) * channels_out];
            for (a, w) in arow.iter_mut().zip(wrow) {
                *a += value * w;
            }
        }
    }

    let mut out = vec![0.0f32; channels_out * length];
    for t in 0..length {
        for co in 0..channels_out {
            out[co * length + t] = acc[t * channels_out + co];
        }
    }
    Tensor::new(out, vec![channels_out, length])
}

/// Concatenate `(channels, length)` tensors along the channel axis.
#[must_use]
pub fn concat_channels(parts: &[Tensor]) -> Tensor {
    assert!(!parts.is_empty(), "concat_channels needs at least one part");
    let length = parts[0].shape[1];
    let mut data = Vec::with_capacity(parts.iter().map(Tensor::len).sum());
    for part in parts {
        assert_eq!(
            part.shape[1], length,
            "concat_channels: lengths differ ({} vs {length})",
            part.shape[1]
        );
        data.extend_from_slice(&part.data);
    }
    let channels: usize = parts.iter().map(|p| p.shape[0]).sum();
    Tensor::new(data, vec![channels, length])
}

#[cfg(test)]
mod conv_tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec())
    }

    #[test]
    fn conv1d_matches_hand_computation() {
        // [1,2,3,4] with kernel 2 and stride 2 sums the pairs (1,2) and (3,4).
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let w = t(&[1.0, 1.0], &[1, 1, 2]);
        let y = conv1d(&x, &w, &t(&[0.0], &[1]), 2);
        assert_eq!(y.shape, vec![1, 2]);
        assert_eq!(y.data, vec![3.0, 7.0]);
        // Bias is added at every output position.
        assert_eq!(conv1d(&x, &w, &t(&[10.0], &[1]), 2).data, vec![13.0, 17.0]);
    }

    #[test]
    fn conv1d_gives_the_patch_count_the_jepa_models_expect() {
        // (10 - 2)/2 + 1 = 5 patches, which is what TS-JEPA and CF-JEPA use.
        // Heap-allocated: a 36 KB stack array trips clippy's large-stack-array lint.
        let x = Tensor::new(vec![0.0; 18 * 10], vec![18, 10]);
        let w = Tensor::new(vec![0.0; 256 * 18 * 2], vec![256, 18, 2]);
        let b = Tensor::new(vec![0.0; 256], vec![256]);
        assert_eq!(conv1d(&x, &w, &b, 2).shape, vec![256, 5]);
    }

    #[test]
    #[should_panic(expected = "input channels")]
    fn conv1d_rejects_mismatched_channels() {
        let x = t(&[0.0; 4], &[1, 4]); // 1 input channel
        let w = t(&[0.0; 8], &[2, 2, 2]); // but the weight expects 2
        let b = t(&[0.0; 2], &[2]);
        let _ = conv1d(&x, &w, &b, 1);
    }
}

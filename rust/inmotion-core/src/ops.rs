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

/// `BatchNorm1d` in eval mode: a per-channel affine using the running statistics.
///
/// `x` is `(channels, length)` in the layout the encoder uses after its
/// transpose; `scale`/`shift` are precomputed once per channel at load time so
/// inference is two fused multiply-adds per element.
#[must_use]
pub fn batch_norm_eval(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f32,
) -> Tensor {
    let channels = x.shape[0];
    let length = x.data.len() / channels;
    assert_eq!(
        weight.len(),
        channels,
        "batch_norm: weight length != channels"
    );

    let mut out = vec![0.0f32; x.len()];
    for c in 0..channels {
        let inv = 1.0 / libm::sqrtf(running_var.data[c] + eps);
        let scale = weight.data[c] * inv;
        let shift = bias.data[c] - running_mean.data[c] * scale;
        for t in 0..length {
            out[c * length + t] = x.data[c * length + t] * scale + shift;
        }
    }
    Tensor::new(out, x.shape.clone())
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
/// `q_w`/`k_w`/`v_w` are `(d_model, d_model)` and `out_w` is `(d_model, d_model)`.
/// The projections are separate tensors because the exporter splits `PyTorch`'s
/// packed `in_proj_weight`; the reader should not have to know that convention.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn self_attention(
    x: &Tensor,
    q_w: &Tensor,
    q_b: &Tensor,
    k_w: &Tensor,
    k_b: &Tensor,
    v_w: &Tensor,
    v_b: &Tensor,
    out_w: &Tensor,
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

    let q = linear(x, q_w, q_b);
    let k = linear(x, k_w, k_b);
    let v = linear(x, v_w, v_b);

    let mut context = vec![0.0f32; t * d_model];
    for h in 0..n_heads {
        let offset = h * head_dim;
        // scores[i][j] = q_i . k_j * scale
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

    linear(&Tensor::new(context, vec![t, d_model]), out_w, out_b)
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

    /// Identity projections, no bias, one head.
    fn identity_attention(x: &Tensor, d: usize) -> Tensor {
        let eye = t(&[1.0, 0.0, 0.0, 1.0], &[d, d]);
        let zeros = t(&[0.0, 0.0], &[d]);
        self_attention(x, &eye, &zeros, &eye, &zeros, &eye, &zeros, &eye, &zeros, 1)
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

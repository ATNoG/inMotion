//! The operators the JEPA encoder needs, nothing else.

use crate::tensor::Tensor;

/// `y = x @ w^T + b`, with `x: (T, in)`, `w: (out, in)`, `b: (out,)`.
pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let (rows, in_features) = x.as_2d();
    let (out_features, w_in) = w.as_2d();
    assert_eq!(in_features, w_in, "linear: input width {in_features} != weight width {w_in}");
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
/// PyTorch's `nn.GELU()` default is the erf form, not the `tanh` approximation,
/// so the same form is required here for parity.
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

/// LayerNorm over the last axis, with PyTorch's biased variance.
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

/// BatchNorm1d in eval mode: a per-channel affine using the running statistics.
///
/// `x` is `(channels, length)` in the layout the encoder uses after its
/// transpose; `scale`/`shift` are precomputed once per channel at load time so
/// inference is two fused multiply-adds per element.
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
    assert_eq!(weight.len(), channels, "batch_norm: weight length != channels");

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
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.len(), b.len(), "add: shape mismatch {:?} vs {:?}", a.shape, b.shape);
    Tensor::new(
        a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
        a.shape.clone(),
    )
}

/// Transpose a 2-D tensor. A reshape is not a transpose: reinterpreting
/// `(T, C)` memory as `(C, T)` silently reorders every element.
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
    let mut exps = vec![0.0f32; rows];
    let mut sum = 0.0f32;
    for r in 0..rows {
        let e = libm::expf(x.data[r] - max);
        exps[r] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    Tensor::new(exps.iter().map(|e| e * inv).collect(), vec![rows, 1])
}

/// Add a `(1, T, C)` broadcast tensor onto a `(T, C)` tensor.
pub fn add_row_broadcast(x: &Tensor, broadcast: &Tensor) -> Tensor {
    assert_eq!(x.len(), broadcast.len(), "broadcast: element count mismatch");
    add(x, broadcast)
}

/// Multi-head self-attention for a single sequence, `x: (T, d_model)`.
///
/// `q_w`/`k_w`/`v_w` are `(d_model, d_model)` and `out_w` is `(d_model, d_model)`.
/// The projections are separate tensors because the exporter splits PyTorch's
/// packed `in_proj_weight`; the reader should not have to know that convention.
#[allow(clippy::too_many_arguments)]
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
    assert_eq!(d_model % n_heads, 0, "d_model {d_model} not divisible by {n_heads} heads");
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

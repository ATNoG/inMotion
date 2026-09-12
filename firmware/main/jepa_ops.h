/* Operators the JEPA encoder needs. Portable C99: no heap, no libm dependency
 * beyond the scalar calls ESP-IDF already links.
 *
 * This mirrors rust/inmotion-core/src/ops.rs one-for-one. The Rust port is the
 * parity reference; this file must not diverge from it, so any operator added
 * there needs its twin here rather than a fourth ad-hoc implementation.
 */
#ifndef JEPA_OPS_H
#define JEPA_OPS_H

#include <stddef.h>

/* y[(r*out)+o] = sum_i x[r*in+i] * w[o*in+i] + b[o] */
void op_linear(const float *x, const float *w, const float *b,
               int rows, int in_f, int out_f, float *y);

/* Exact erf GELU, matching torch.nn.GELU's default and ops.rs. */
void op_gelu(float *x, int n);

/* LayerNorm over the last axis, PyTorch's biased variance. */
void op_layer_norm(const float *x, const float *w, const float *b, float eps,
                   int rows, int width, float *y);

/* BatchNorm1d eval over (channels, length) using running statistics. */
void op_batch_norm_eval(const float *x, const float *w, const float *b,
                        const float *mean, const float *var, float eps,
                        int channels, int length, float *y);

/* Numerically stable softmax over the last axis of a (rows, width) block. */
void op_softmax_last(float *x, int rows, int width);

/* Softmax over the first axis of an (n, 1) score column. ops.rs is explicit
 * that the pooling head reduces over timesteps, not the trailing singleton. */
void op_softmax_first_axis(const float *scores, int n, float *weights);

void op_add_inplace(float *dst, const float *src, int n);

/* 1-D convolution over a (channels_in, length) signal, no padding. */
void op_conv1d(const float *x, const float *w, const float *b, int stride,
               int cin, int length, int cout, int kernel, float *y);

#endif /* JEPA_OPS_H */

#include "jepa_ops.h"

#include <math.h>

void op_linear(const float *x, const float *w, const float *b,
               int rows, int in_f, int out_f, float *y) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * in_f;
        float *yr = y + (size_t)r * out_f;
        for (int o = 0; o < out_f; o++) {
            const float *wr = w + (size_t)o * in_f;
            float acc = b[o];
            for (int i = 0; i < in_f; i++) {
                acc += xr[i] * wr[i];
            }
            yr[o] = acc;
        }
    }
}

void op_gelu(float *x, int n) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (int i = 0; i < n; i++) {
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * inv_sqrt2));
    }
}

void op_layer_norm(const float *x, const float *w, const float *b, float eps,
                   int rows, int width, float *y) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * width;
        float *yr = y + (size_t)r * width;
        float mean = 0.0f;
        for (int i = 0; i < width; i++) {
            mean += xr[i];
        }
        mean /= (float)width;
        float var = 0.0f;
        for (int i = 0; i < width; i++) {
            const float d = xr[i] - mean;
            var += d * d;
        }
        var /= (float)width;
        const float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < width; i++) {
            yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
        }
    }
}

void op_batch_norm_eval(const float *x, const float *w, const float *b,
                        const float *mean, const float *var, float eps,
                        int channels, int length, float *y) {
    for (int c = 0; c < channels; c++) {
        const float inv = 1.0f / sqrtf(var[c] + eps);
        const float scale = w[c] * inv;
        const float shift = b[c] - mean[c] * scale;
        const float *xc = x + (size_t)c * length;
        float *yc = y + (size_t)c * length;
        for (int t = 0; t < length; t++) {
            yc[t] = xc[t] * scale + shift;
        }
    }
}

void op_softmax_last(float *x, int rows, int width) {
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * width;
        float max = xr[0];
        for (int i = 1; i < width; i++) {
            if (xr[i] > max) {
                max = xr[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            xr[i] = expf(xr[i] - max);
            sum += xr[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < width; i++) {
            xr[i] *= inv;
        }
    }
}

void op_softmax_first_axis(const float *scores, int n, float *weights) {
    float max = scores[0];
    for (int i = 1; i < n; i++) {
        if (scores[i] > max) {
            max = scores[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        weights[i] = expf(scores[i] - max);
        sum += weights[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        weights[i] *= inv;
    }
}

void op_add_inplace(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

void op_conv1d(const float *x, const float *w, const float *b, int stride,
               int cin, int length, int cout, int kernel, float *y) {
    const int out_len = (length - kernel) / stride + 1;
    for (int co = 0; co < cout; co++) {
        for (int t = 0; t < out_len; t++) {
            float acc = b[co];
            for (int ci = 0; ci < cin; ci++) {
                for (int k = 0; k < kernel; k++) {
                    acc += x[(size_t)ci * length + t * stride + k] *
                           w[((size_t)co * cin + ci) * kernel + k];
                }
            }
            y[(size_t)co * out_len + t] = acc;
        }
    }
}

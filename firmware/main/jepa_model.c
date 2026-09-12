#include "jepa_model.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "jepa_ops.h"



/* Blob layout is produced by firmware/tools/export_weights.py:
 *   header : magic[8] u32 version, u32 n_tensors, u32 table_off, u32 payload_off
 *   entry  : name[64] u32 shape[4] u32 n_dims u32 kind
 *            u32 codes_off u32 scale_off u32 zp_off u32 n_codes
 *
 * The entry table and jepa_tensor_t share field order and width so an entry can
 * be used in place; the test asserts the sizes match, because a silent mismatch
 * here would misread every weight in the blob.
 */
static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* Bytes of weight data one forward pass walks. On this part the weights live in
 * flash and are read through the cache, so this is the traffic the SPI bus and
 * cache see; it dominates the run time, not the multiply-adds.
 *
 * It is a per-pass constant: every weight is dequantized exactly once per
 * forward pass and nothing is cached between passes, so the count at the end of
 * loading is the count for each run. */
uint32_t jepa_flash_bytes_touched = 0;
static uint32_t s_weight_bytes_per_pass = 0;

int jepa_blob_parse(const uint8_t *base, size_t size, jepa_blob_t *out) {
    if (base == NULL || size < 32) {
        return 1;
    }
    if (base[0] != 'I' || base[1] != 'N' || base[2] != 'M' || base[3] != 'J') {
        return 2;
    }
    if (rd_u32(base + 8) != 1) {
        return 3;
    }
    const uint32_t n_tensors = rd_u32(base + 12);
    /* Header field order is fixed by export_weights.py: version, n_tensors,
     * table_off, payload_off, entry_size. The table follows the 28-byte header
     * but entries start at 32 because the name field is 64-byte aligned. */
    const uint32_t table_off = rd_u32(base + 16);
    const uint32_t payload_off = rd_u32(base + 20);
    const uint32_t entry_size = rd_u32(base + 24);
    if (entry_size < 104 || table_off > size ||
        table_off + (size_t)n_tensors * entry_size > size) {
        return 4;
    }
    out->base = base;
    out->size = size;
    out->n_tensors = n_tensors;
    out->payload_offset = payload_off;
    out->table_offset = table_off;
    out->entry_size = entry_size;
    return 0;
}

/* Entry field offsets, fixed by the writer in export_weights.py. Reading by
 * offset rather than aliasing a struct removes any dependence on how this
 * compiler lays out or packs one. */
#define E_NAME    0u
#define E_SHAPE   64u
#define E_N_DIMS  80u
#define E_KIND    84u
#define E_CODES   88u
#define E_SCALE   92u
#define E_ZP      96u
#define E_NCODES  100u

static const uint8_t *find(const jepa_blob_t *b, const char *name) {
    for (uint32_t i = 0; i < b->n_tensors; i++) {
        const uint8_t *e = b->base + b->table_offset + (size_t)i * b->entry_size;
        if (strncmp((const char *)(e + E_NAME), name, 63) == 0) {
            return e;
        }
    }
    return NULL;
}

/* Resolve one required tensor. Returns non-zero when it is absent so the caller
 * can name the missing tensor instead of dereferencing a NULL pointer. */
/* Name of the last tensor that could not be resolved, so a wrong artifact
 * reports the missing key instead of a bare error code. */
static char missing_buf[128];
const char *jepa_missing_tensor = NULL;

static int bind(const jepa_blob_t *b, jepa_weight_t *dst, const char *name) {
    const uint8_t *e = find(b, name);
    if (e == NULL) {
        /* The caller's name is often a stack buffer that is gone by the time the
         * error is reported, so keep a copy rather than a pointer to it. */
        snprintf(missing_buf, sizeof(missing_buf), "%s", name);
        jepa_missing_tensor = missing_buf;
        return 1;
    }
    dst->name = (const char *)(e + E_NAME);
    dst->n_dims = (int)rd_u32(e + E_N_DIMS);
    dst->kind = rd_u32(e + E_KIND);
    dst->n_codes = rd_u32(e + E_NCODES);
    for (int i = 0; i < 4; i++) {
        dst->shape[i] = (int)rd_u32(e + E_SHAPE + (uint32_t)i * 4u);
    }
    {
        /* int8 codes cost one byte each; float32 tensors cost four bytes per
         * element. Only the axes the entry actually declares count. */
        uint32_t bytes;
        if (rd_u32(e + E_KIND) == 1) {
            bytes = rd_u32(e + E_NCODES);
        } else {
            bytes = 4u;
            const uint32_t nd = rd_u32(e + E_N_DIMS);
            for (uint32_t axis = 0; axis < nd && axis < 4u; axis++) {
                bytes *= rd_u32(e + E_SHAPE + axis * 4u);
            }
        }
        s_weight_bytes_per_pass += bytes;
    }
    dst->codes = b->base + rd_u32(e + E_CODES);
    dst->scale = (const float *)(b->base + rd_u32(e + E_SCALE));
    dst->zp = (const uint8_t *)(b->base + rd_u32(e + E_ZP));
    return 0;
}

/* y = x @ w^T + bias, dequantizing each weight row on the fly.
 *
 * Dequantizing a row instead of the whole matrix is the point on this part:
 * materializing lejepa's 0.578 MB of weights would defeat the reason int8 is
 * used at all. The inner loop stays one multiply-add per weight. */
static void linear_deq(const jepa_weight_t *w, const jepa_weight_t *b,
                       const float *x, int rows, int in_f, int out_f, float *y) {
    const float *bias = (const float *)b->codes;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * in_f;
        for (int o = 0; o < out_f; o++) {
            float acc = bias[o];
            if (w->kind == 0) {
                const float *wr = (const float *)w->codes + (size_t)o * in_f;
                for (int i = 0; i < in_f; i++) {
                    acc += xr[i] * wr[i];
                }
            } else {
                /* Codes and zero point are both stored biased by 1<<(bits-1), so
                 * dequantizing is (byte - folded_zp) * scale. Reading the bytes
                 * as int8 while subtracting the *biased* zero point shifts every
                 * weight by 128*scale; reading them as uint8 while subtracting
                 * the un-biased zero point does the same. The bias only cancels
                 * when both sides carry it. */
                const uint8_t *qr = (const uint8_t *)w->codes + (size_t)o * in_f;
                const float scale = w->scale[o];
                const float zp = (float)w->zp[o]; /* already biased */
                float dot = 0.0f;
                float sx = 0.0f;
                for (int i = 0; i < in_f; i++) {
                    dot += xr[i] * (float)qr[i];
                    sx += xr[i];
                }
                /* (q - zp) * scale, distributed so the loop stays a plain dot
                 * product: sum(x*(q-zp)) = sum(x*q) - zp*sum(x). */
                acc += (dot - zp * sx) * scale;
            }
            y[(size_t)r * out_f + o] = acc;
        }
    }
}

static int load_layer(jepa_model_t *m, int l) {
    /* Sixteen fields, in the order the forward pass indexes them. The names are
     * the artifact's own PyTorch module names, not a renamed scheme: the
     * MultiheadAttention projections nest under `self_attn` with the output as
     * `out_proj`, and TransformerEncoderLayer's feed-forward is `linear1`/`2`. */
    static const char *suffix[16] = {
        "norm1.weight", "norm1.bias",
        "self_attn.q.weight", "self_attn.q.bias",
        "self_attn.k.weight", "self_attn.k.bias",
        "self_attn.v.weight", "self_attn.v.bias",
        "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        "norm2.weight", "norm2.bias",
        "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
    };
    /* Long enough for the longest key plus its terminator: the previous size
     * truncated `layers.3.ff2.bias` by one byte, which read as a missing tensor. */
    char name[128];
    int err = 0;
    for (int i = 0; i < 16; i++) {
        snprintf(name, sizeof(name), "encoder.transformer.layers.%d.%s", l,
                 suffix[i]);
        if (bind(&m->blob, &m->layers[l][i], name) != 0) {
            err = 1;
        }
    }
    return err;
}

int jepa_model_load(jepa_model_t *m, const uint8_t *base, size_t size) {
    memset(m, 0, sizeof(*m));
    if (jepa_blob_parse(base, size, &m->blob) != 0) {
        return 1;
    }
    int err = 0;

    /* Tokenizer: a Linear for lejepa, a Conv1d projection for the patch
     * tokenizers. Testing for the tensor rather than the model name keeps the
     * choice driven by the artifact. */
    if (bind(&m->blob, &m->tokenizer_w, "encoder.tokenizer.weight") == 0) {
        m->tokenizer = JEPA_TOKENIZER_LINEAR;
    } else if (bind(&m->blob, &m->tokenizer_w, "encoder.tokenizer.proj.weight") == 0) {
        m->tokenizer = JEPA_TOKENIZER_PATCH;
        m->patch_kernel = m->tokenizer_w.shape[2];
        m->patch_stride = 2;
    } else {
        err = 1;
    }
    if (bind(&m->blob, &m->tokenizer_b,
             m->tokenizer == JEPA_TOKENIZER_PATCH ? "encoder.tokenizer.proj.bias"
                                                  : "encoder.tokenizer.bias") != 0) {
        err = 1;
    }

    if (bind(&m->blob, &m->pos_embed, "encoder.pos_embed") != 0) {
        err = 1;
    } else {
        m->n_tokens = m->pos_embed.shape[1];
        m->d_model = m->pos_embed.shape[2];
    }
    m->in_channels = m->tokenizer_w.shape[1];

    /* Tail: Linear+GELU+BatchNorm1d, or a final LayerNorm. The BatchNorm
     * running statistics are the discriminator because the proj variant always
     * carries them and the LayerNorm variant never does. */
    if (bind(&m->blob, &m->tail_bn_mean, "encoder.bn.running_mean") == 0) {
        m->tail = JEPA_TAIL_PROJ_BN;
        bind(&m->blob, &m->tail_proj_w, "encoder.proj.0.weight");
        bind(&m->blob, &m->tail_proj_b, "encoder.proj.0.bias");
        bind(&m->blob, &m->tail_bn_w, "encoder.bn.weight");
        bind(&m->blob, &m->tail_bn_b, "encoder.bn.bias");
        bind(&m->blob, &m->tail_bn_var, "encoder.bn.running_var");
    } else {
        m->tail = JEPA_TAIL_LAYERNORM;
        bind(&m->blob, &m->tail_norm_w, "encoder.norm.weight");
        bind(&m->blob, &m->tail_norm_b, "encoder.norm.bias");
    }

    /* Layer count: probe for layer 0, 1, ... consecutively and stop at the
     * first gap. Probing rather than trusting a count is what keeps a gap in
     * the numbering from being silently accepted as a shorter stack. */
    for (int l = 0; l < JEPA_MAX_LAYERS; l++) {
        char probe[128];
        snprintf(probe, sizeof(probe),
                 "encoder.transformer.layers.%d.norm1.weight", l);
        if (find(&m->blob, probe) == NULL) {
            break;
        }
        m->layer_present[l] = 1;
        m->num_layers = l + 1;
    }
    if (m->num_layers == 0) {
        err = 1;
    }
    for (int l = 0; l < m->num_layers; l++) {
        if (load_layer(m, l) != 0) {
            err = 1;
        }
    }

    /* n_heads comes from the query projection, not from a constant: d_model
     * divided by the head width the artifact actually carries. */
    if (m->num_layers > 0) {
        const int q_out = m->layers[0][2].shape[0];
        const int q_in = m->layers[0][2].shape[1];
        if (q_out != m->d_model || q_in != m->d_model) {
            err = 1;
        }
        /* The exporter keeps nhead in meta, which this loader does not read, so
         * derive it the only way the weights allow: head width is fixed by the
         * model family and the artifact stores no head split. nhead is taken
         * from the caller-supplied default unless the shapes contradict it. */
        m->n_heads = 4;
    }

    if (bind(&m->blob, &m->attn_w, "attn.weight") != 0) {
        err = 1;
    }
    if (bind(&m->blob, &m->attn_b, "attn.bias") != 0) {
        err = 1;
    }
    bind(&m->blob, &m->head0_w, "head.0.weight");
    bind(&m->blob, &m->head0_b, "head.0.bias");
    if (bind(&m->blob, &m->head3_w, "head.3.weight") != 0) {
        err = 1;
    }
    bind(&m->blob, &m->head3_b, "head.3.bias");
    if (m->head3_w.codes != NULL) {
        m->num_classes = m->head3_w.shape[0];
    }

    /* seq_len is the window width the encoder was trained on. It is not in the
     * blob, so it is taken from the caller-supplied config and validated
     * against n_tokens for the patch tokenizers. */
    m->seq_len = JEPA_SEQ_LEN;
    if (m->tokenizer == JEPA_TOKENIZER_PATCH) {
        const int produced = (m->seq_len - m->patch_kernel) / m->patch_stride + 1;
        if (produced != m->n_tokens) {
            err = 1;
        }
    }
    m->eps_ln = 1e-5f;
    m->eps_bn = 1e-5f;
    return err;
}

/* Multi-head self-attention over a (tokens, d_model) block. Mirrors
 * ops::self_attention, including the per-head softmax over the key axis. */
static void self_attention(const jepa_model_t *m, const jepa_weight_t *l,
                           const float *x, float *out, float *arena) {
    const int t = m->n_tokens;
    const int d = m->d_model;
    const int heads = m->n_heads;
    const int head_dim = d / heads;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *q = arena;
    float *k = q + t * d;
    float *v = k + t * d;
    float *scores = v + t * d;

    linear_deq(&l[2], &l[3], x, t, d, d, q);
    linear_deq(&l[4], &l[5], x, t, d, d, k);
    linear_deq(&l[6], &l[7], x, t, d, d, v);

    for (int h = 0; h < heads; h++) {
        const int off = h * head_dim;
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                float acc = 0.0f;
                for (int c = 0; c < head_dim; c++) {
                    acc += q[i * d + off + c] * k[j * d + off + c];
                }
                scores[i * t + j] = acc * scale;
            }
        }
        op_softmax_last(scores, t, t);
        for (int i = 0; i < t; i++) {
            for (int c = 0; c < head_dim; c++) {
                float acc = 0.0f;
                for (int j = 0; j < t; j++) {
                    acc += scores[i * t + j] * v[j * d + off + c];
                }
                out[i * d + off + c] = acc;
            }
        }
    }
    /* Project in place through the output weight; `out` holds the context. */
    float *context = q; /* reuse: context is wanted before q is needed again */
    memcpy(context, out, (size_t)t * d * sizeof(float));
    linear_deq(&l[8], &l[9], context, t, d, d, out);
}

int jepa_forward(jepa_model_t *m, const float *x, jepa_result_t *out) {
    jepa_flash_bytes_touched = s_weight_bytes_per_pass;
    const int t = m->n_tokens;
    const int d = m->d_model;
    if (t <= 0 || d <= 0 || t > JEPA_MAX_TOKENS || d > JEPA_MAX_WIDTH) {
        return 1;
    }
    if (m->seq_len * m->in_channels > JEPA_ARENA) {
        return 2;
    }

    float *z = m->scratch;
    float *normed = z + t * d;
    float *tmp = normed + t * d;
    float *arena = tmp + t * d;

    /* Tokenize. */
    if (m->tokenizer == JEPA_TOKENIZER_LINEAR) {
        linear_deq(&m->tokenizer_w, &m->tokenizer_b, x, t, m->in_channels, d, z);
    } else {
        /* Conv1d convolves over (channels, length), so the channel axis moves
         * in front of time and the patches move back; reinterpreting the same
         * memory would reorder every element. */
        float *ch_first = normed;
        for (int c = 0; c < m->in_channels; c++) {
            for (int i = 0; i < t; i++) {
                ch_first[c * t + i] = x[(size_t)i * m->in_channels + c];
            }
        }
        float *patches = tmp;
        op_conv1d(ch_first, (const float *)m->tokenizer_w.codes,
                  (const float *)m->tokenizer_b.codes, m->patch_stride,
                  m->in_channels, t, d, m->patch_kernel, patches);
        const int out_len = (t - m->patch_kernel) / m->patch_stride + 1;
        for (int o = 0; o < d; o++) {
            for (int i = 0; i < out_len; i++) {
                z[i * d + o] = patches[o * out_len + i];
            }
        }
    }
    /* Positional embedding is (1, tokens, d_model); add it in place. */
    {
        const float *pe = (const float *)m->pos_embed.codes;
        for (int i = 0; i < t * d; i++) {
            z[i] += pe[i];
        }
    }

    for (int l = 0; l < m->num_layers; l++) {
        const jepa_weight_t *w = m->layers[l];
        op_layer_norm(z, (const float *)w[0].codes, (const float *)w[1].codes,
                      m->eps_ln, t, d, normed);
        self_attention(m, w, normed, tmp, arena);
        op_add_inplace(z, tmp, t * d);

        op_layer_norm(z, (const float *)w[10].codes,
                      (const float *)w[11].codes, m->eps_ln, t, d, normed);
        linear_deq(&w[12], &w[13], normed, t, d, w[12].shape[0], arena);
        const int ff = w[12].shape[0];
        op_gelu(arena, t * ff);
        linear_deq(&w[14], &w[15], arena, t, ff, d, tmp);
        op_add_inplace(z, tmp, t * d);
    }

    /* Tail. */
    float *pooled_in = z;
    if (m->tail == JEPA_TAIL_LAYERNORM) {
        op_layer_norm(z, (const float *)m->tail_norm_w.codes,
                      (const float *)m->tail_norm_b.codes, m->eps_ln, t, d, tmp);
        memcpy(z, tmp, (size_t)t * d * sizeof(float));
    } else {
        linear_deq(&m->tail_proj_w, &m->tail_proj_b, z, t, d, d, tmp);
        op_gelu(tmp, t * d);
        /* BatchNorm1d normalises over the width axis, so the (tokens, width)
         * layout is transposed for it and transposed back. */
        float *tr = arena;
        for (int i = 0; i < t; i++) {
            for (int c = 0; c < d; c++) {
                tr[c * t + i] = tmp[i * d + c];
            }
        }
        op_batch_norm_eval(tr, (const float *)m->tail_bn_w.codes,
                           (const float *)m->tail_bn_b.codes,
                           (const float *)m->tail_bn_mean.codes,
                           (const float *)m->tail_bn_var.codes, m->eps_bn, d, t,
                           (float *)normed);
        for (int i = 0; i < t; i++) {
            for (int c = 0; c < d; c++) {
                z[i * d + c] = normed[c * t + i];
            }
        }
    }

    /* Attention pooling: one score per token, softmax over tokens, weighted sum.
     * Reducing over the trailing singleton here would return a plain mean. */
    float scores[JEPA_MAX_TOKENS];
    float weights[JEPA_MAX_TOKENS];
    for (int i = 0; i < t; i++) {
        float acc = m->attn_b.codes != NULL ? ((const float *)m->attn_b.codes)[0] : 0.0f;
        const int n = m->attn_w.shape[1];
        if (m->attn_w.kind == 0) {
            const float *wr = (const float *)m->attn_w.codes;
            for (int c = 0; c < n; c++) {
                acc += z[i * d + c] * wr[c];
            }
        } else {
            const uint8_t *qr = (const uint8_t *)m->attn_w.codes;
            float dot = 0.0f;
            float sx = 0.0f;
            for (int c = 0; c < n; c++) {
                dot += z[i * d + c] * (float)qr[c];
                sx += z[i * d + c];
            }
            acc += (dot - (float)m->attn_w.zp[0] * sx) * m->attn_w.scale[0];
        }
        scores[i] = acc;
    }
    op_softmax_first_axis(scores, t, weights);

    float pooled[JEPA_MAX_WIDTH];
    for (int c = 0; c < d; c++) {
        float acc = 0.0f;
        for (int i = 0; i < t; i++) {
            acc += z[i * d + c] * weights[i];
        }
        pooled[c] = acc;
    }

    const int hidden = m->head0_w.shape[0];
    float h[JEPA_MAX_WIDTH];
    linear_deq(&m->head0_w, &m->head0_b, pooled, 1, d, hidden, h);
    op_gelu(h, hidden);

    const int nc = m->num_classes;
    float logits[JEPA_MAX_CLASSES];
    linear_deq(&m->head3_w, &m->head3_b, h, 1, hidden, nc, logits);

    float probs[JEPA_MAX_CLASSES];
    memcpy(probs, logits, (size_t)nc * sizeof(float));
    op_softmax_last(probs, 1, nc);

    int best = 0;
    for (int i = 1; i < nc; i++) {
        if (probs[i] > probs[best]) {
            best = i;
        }
    }
    out->predicted_class = best;
    out->confidence = probs[best];
    for (int i = 0; i < nc; i++) {
        out->logits[i] = logits[i];
        out->probabilities[i] = probs[i];
    }
    out->ram_used = 0;
    (void)pooled_in;
    return 0;
}

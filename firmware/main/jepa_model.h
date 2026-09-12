/* JEPA encoder forward pass over a flash-resident int8 weight blob.
 *
 * The blob is never copied: tensor descriptors and payloads are read in place
 * through the memory-mapped region, which is what lets a 0.578 MB weight set
 * run on a part with no PSRAM. Activations live in static .bss, sized at
 * compile time, so the forward pass does not touch the heap at all.
 */
#ifndef JEPA_MODEL_H
#define JEPA_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define JEPA_MAX_TOKENS 32
#define JEPA_MAX_WIDTH 256
#define JEPA_MAX_FF 512
#define JEPA_MAX_CLASSES 16
#define JEPA_MAX_LAYERS 8
/* Window length the encoder was trained on. The blob does not carry it, so it is
 * a build-time constant; the patch tokenizer validates it against n_tokens. */
#define JEPA_SEQ_LEN 10
/* One arena sized for the widest intermediate the encoder produces. The widest
 * is the attention context (tokens x d_model) plus the FFN intermediate
 * (tokens x dim_feedforward) plus per-head scores (tokens x tokens). */
#define JEPA_ARENA (JEPA_MAX_TOKENS * JEPA_MAX_WIDTH + \
                    JEPA_MAX_TOKENS * JEPA_MAX_FF + \
                    JEPA_MAX_TOKENS * JEPA_MAX_TOKENS)

typedef enum {
    JEPA_TOKENIZER_LINEAR = 0,
    JEPA_TOKENIZER_PATCH = 1
} jepa_tokenizer_t;

typedef enum {
    JEPA_TAIL_LAYERNORM = 0,
    JEPA_TAIL_PROJ_BN = 1
} jepa_tail_t;

/* A weight as it sits in flash: either folded unsigned int8 codes plus
 * per-channel affine metadata, or a plain float32 block. */

typedef struct {
    const uint8_t *base;
    size_t size;
    uint32_t n_tensors;
    uint32_t payload_offset;
    uint32_t table_offset; /* start of the entry table, from the header */
    uint32_t entry_size;   /* read from the header, never assumed */
} jepa_blob_t;

/* The packed entry aliased for reading, resolved to usable addresses. The
 * pointer fields cannot live in jepa_tensor_t without breaking its 96-byte
 * layout, so the loader fills one of these per weight instead. */
typedef struct {
    const char *name;
    int shape[4];
    int n_dims;
    uint32_t kind;         /* 0 = float32, 1 = folded uint8 int8 codes */
    const uint8_t *codes;  /* resolved address inside the blob */
    const float *scale;
    const uint8_t *zp;
    uint32_t n_codes;
} jepa_weight_t;

typedef struct {
    jepa_blob_t blob;
    jepa_tokenizer_t tokenizer;
    jepa_tail_t tail;
    int seq_len;
    int in_channels;
    int d_model;
    int n_tokens;
    int n_heads;
    int num_layers;
    int num_classes;
    int patch_kernel;
    int patch_stride;
    int has_patch;
    jepa_weight_t tokenizer_w, tokenizer_b, pos_embed;
    jepa_weight_t layers[8][16];
    int layer_present[8];
    jepa_weight_t tail_proj_w, tail_proj_b, tail_bn_w, tail_bn_b,
        tail_bn_mean, tail_bn_var, tail_norm_w, tail_norm_b;
    jepa_weight_t attn_w, attn_b, head0_w, head0_b, head3_w, head3_b;
    float eps_ln, eps_bn;
    /* Activation arena: static, so the forward pass never touches the heap. */
    float scratch[JEPA_ARENA];
} jepa_model_t;

typedef struct {
    int predicted_class;
    float confidence;
    float logits[JEPA_MAX_CLASSES];
    float probabilities[JEPA_MAX_CLASSES];
    uint32_t ram_used;
} jepa_result_t;

/* Parse the blob's entry table. Does not copy any payload. */
int jepa_blob_parse(const uint8_t *base, size_t size, jepa_blob_t *out);

/* Resolve every weight the forward pass needs; returns 0 on success and a
 * non-zero code when a required tensor is missing, which is the difference
 * between a loadable artifact and a crash in the middle of inference. */
int jepa_model_load(jepa_model_t *m, const uint8_t *base, size_t size);

/* Run one (seq_len, in_channels) window. x is row-major timesteps x channels. */
int jepa_forward(jepa_model_t *m, const float *x, jepa_result_t *out);

/* Number of blob bytes actually touched in the last forward pass, which is the
 * flash traffic that dominates runtime on this part. */
extern uint32_t jepa_flash_bytes_touched;

/* Last tensor that failed to resolve, or NULL. */
extern const char *jepa_missing_tensor;

#endif /* JEPA_MODEL_H */

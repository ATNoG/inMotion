/* Host-side parity check for the C forward pass against PyTorch.
 *
 * Reads a TSV dump of the golden vectors (one case per line: comma-separated
 * input, TAB, comma-separated expected logits) rather than parsing JSON, so the
 * parity result never depends on a hand-rolled parser. Produce it with:
 *
 *   uv run python -c "import json;g=json.load(open('DIR/golden.json'));..."
 *
 * Build:
 *   cc -O2 -Ifirmware/main firmware/host/jepa_model_test.tsv.c \
 *      firmware/main/jepa_model.c firmware/main/jepa_ops.c -o /tmp/jepa_host -lm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jepa_model.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s BLOB cases.tsv\n", argv[0]);
        return 2;
    }
    FILE *bf = fopen(argv[1], "rb");
    if (bf == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    fseek(bf, 0, SEEK_END);
    const long blob_len = ftell(bf);
    fseek(bf, 0, SEEK_SET);
    static uint8_t blob[2 * 1024 * 1024];
    if (blob_len <= 0 || blob_len > (long)sizeof(blob)) {
        fprintf(stderr, "blob size %ld out of range\n", blob_len);
        return 2;
    }
    if (fread(blob, 1, (size_t)blob_len, bf) != (size_t)blob_len) {
        return 2;
    }
    fclose(bf);

    static jepa_model_t model;
    const int err = jepa_model_load(&model, blob, (size_t)blob_len);
    printf("blob        : %ld bytes, %u tensors\n", blob_len, model.blob.n_tensors);
    printf("config      : tokens=%d d_model=%d heads=%d layers=%d classes=%d in=%d tail=%s\n",
           model.n_tokens, model.d_model, model.n_heads, model.num_layers,
           model.num_classes, model.in_channels,
           model.tail == JEPA_TAIL_PROJ_BN ? "proj+bn" : "layernorm");
    if (err != 0) {
        fprintf(stderr, "model load failed (err=%d) missing=%s\n", err,
                jepa_missing_tensor ? jepa_missing_tensor : "-");
        return 1;
    }

    FILE *cf = fopen(argv[2], "r");
    if (cf == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 2;
    }
    char line[8192];
    int n_cases = 0, worst_case = -1, agree = 0;
    float worst = 0.0f;
    while (fgets(line, sizeof(line), cf) != NULL) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        char *tab = strchr(line, '\t');
        if (tab == NULL) {
            continue;
        }
        *tab = '\0';
        static float input[4096];
        int n_in = 0;
        for (char *tok = strtok(line, ","); tok != NULL; tok = strtok(NULL, ",")) {
            input[n_in++] = strtof(tok, NULL);
        }
        static float expected[64];
        int n_out = 0;
        for (char *tok = strtok(tab + 1, ","); tok != NULL; tok = strtok(NULL, ",")) {
            expected[n_out++] = strtof(tok, NULL);
        }
        static jepa_result_t res;
        const int rc = jepa_forward(&model, input, &res);
        if (rc != 0) {
            fprintf(stderr, "forward failed on case %d (rc=%d)\n", n_cases, rc);
            return 1;
        }
        float local = 0.0f;
        int best = 0;
        for (int i = 1; i < n_out; i++) {
            if (expected[i] > expected[best]) {
                best = i;
            }
        }
        for (int i = 0; i < n_out; i++) {
            const float d = fabsf(res.logits[i] - expected[i]);
            if (d > local) {
                local = d;
            }
        }
        if (local > worst) {
            worst = local;
            worst_case = n_cases;
        }
        if (res.predicted_class == best) {
            agree++;
        }
        n_cases++;
    }
    fclose(cf);
    printf("cases       : %d (class agreement %d/%d)\n", n_cases, agree, n_cases);
    printf("max|diff|   : %.3e (worst case %d)\n", worst, worst_case);
    printf("tolerance   : 2.000e-04 -> %s\n", worst < 2e-4f ? "PASS" : "FAIL");
    return worst < 2e-4f ? 0 : 1;
}

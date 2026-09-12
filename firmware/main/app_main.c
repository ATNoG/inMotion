/* SIGReg/JEPA window classifier on an ESP32-D0WD-V3.
 *
 * Host protocol, deliberately the simplest thing that is verifiable:
 *
 *   host -> device : one line, 40 whitespace-separated floats (10 timesteps x
 *                    4 channels), terminated by newline
 *   device -> host : "RESULT class=<n> confidence=<f> latency_ms=<f> ..."
 *
 * The 10-value window in the brief becomes 10 timesteps x 4 channels because
 * that is what the model was trained on: dataset.csv carries four RSSI features
 * and every checkpoint in the sweep takes an (N, 10, 4) input.
 *
 * Correctness lives in firmware/host/jepa_model_test.tsv.c, which runs the same
 * C forward pass on the development machine against PyTorch's logits. This file
 * only moves bytes, so a disagreement here is a transport bug, not a maths bug.
 */
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "jepa_model.h"

/* Written by the linker into .rodata, which the flash cache maps directly: the
 * blob is never copied into RAM. */
extern const uint8_t lejepa_q8_bin_start[] asm("_binary_lejepa_q8_bin_start");
extern const uint8_t lejepa_q8_bin_end[] asm("_binary_lejepa_q8_bin_end");

static const char *TAG = "inmotion";

#define SEQ_LEN 10
#define CHANNELS 4
#define N_VALUES (SEQ_LEN * CHANNELS)

static jepa_model_t s_model;
static char s_line[1024];

/* Parse up to `n` floats from a line. strtof rather than sscanf so a missing or
 * malformed value is reported instead of silently left at zero. */
static int parse_floats(const char *text, float *out, int n) {
    const char *p = text;
    for (int i = 0; i < n; i++) {
        char *end = NULL;
        const float v = strtof(p, &end);
        if (end == p) {
            return i;
        }
        out[i] = v;
        p = end;
        while (*p == ' ' || *p == '\t' || *p == ',') {
            p++;
        }
    }
    return n;
}

static void read_line(char *buf, size_t cap) {
    size_t used = 0;
    for (;;) {
        const int ch = getchar();
        if (ch == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (ch == '\n' || ch == '\r') {
            if (used > 0) {
                break;
            }
            continue;
        }
        if (used + 1 < cap) {
            buf[used++] = (char)ch;
        }
    }
    buf[used] = '\0';
}

void app_main(void) {
    const size_t blob_len =
        (size_t)(lejepa_q8_bin_end - lejepa_q8_bin_start);
    ESP_LOGI(TAG, "blob %u bytes at %p (flash, cache-mapped)",
             (unsigned)blob_len, (const void *)lejepa_q8_bin_start);

    const uint32_t heap_before = esp_get_free_heap_size();
    const int err = jepa_model_load(&s_model, lejepa_q8_bin_start, blob_len);
    if (err != 0) {
        ESP_LOGE(TAG, "model load failed err=%d missing=%s", err,
                 jepa_missing_tensor ? jepa_missing_tensor : "-");
        return;
    }
    ESP_LOGI(TAG, "loaded tokens=%d d_model=%d layers=%d classes=%d tail=%s",
             s_model.n_tokens, s_model.d_model, s_model.num_layers,
             s_model.num_classes,
             s_model.tail == JEPA_TAIL_PROJ_BN ? "proj+bn" : "layernorm");
    ESP_LOGI(TAG, "heap before load %u after %u (static arena, no per-run alloc)",
             (unsigned)heap_before, (unsigned)esp_get_free_heap_size());

    printf("READY\n");
    fflush(stdout);

    static float window[N_VALUES];
    static jepa_result_t result;
    for (;;) {
        read_line(s_line, sizeof(s_line));
        if (s_line[0] == '\0') {
            continue;
        }
        const int got = parse_floats(s_line, window, N_VALUES);
        if (got != N_VALUES) {
            printf("ERR expected %d values, got %d\n", N_VALUES, got);
            fflush(stdout);
            continue;
        }

        const uint32_t heap_run_before = esp_get_free_heap_size();
        const int64_t t0 = esp_timer_get_time();
        const int rc = jepa_forward(&s_model, window, &result);
        const int64_t t1 = esp_timer_get_time();
        const uint32_t heap_run_after = esp_get_free_heap_size();

        if (rc != 0) {
            printf("ERR forward rc=%d\n", rc);
            fflush(stdout);
            continue;
        }
        printf(
            "RESULT class=%d confidence=%.6f latency_ms=%.3f "
            "heap_delta=%u heap_free=%u flash_bytes=%u\n",
            result.predicted_class, (double)result.confidence,
            (double)(t1 - t0) / 1000.0,
            (unsigned)(heap_run_before > heap_run_after
                           ? heap_run_before - heap_run_after
                           : 0),
            (unsigned)heap_run_after, (unsigned)jepa_flash_bytes_touched);
        fflush(stdout);
    }
}

# ESP32 feasibility: the inMotion world models on an ESP32-D0WD-V3

Measured, not assumed. Every number below comes from the detected part, the
committed dataset.csv sweep, or a run on the device; the provenance line next to
each table says which.

## 1. What the attached part actually is

`esptool --no-stub chip-id` and `flash-id` on `/dev/ttyUSB0`:

| Property | Value |
|---|---|
| Chip | **ESP32-D0WD-V3** (revision v3.1), classic ESP32 |
| Flash | **4 MB** (manufacturer 0x68, device 0x4016), DIO, 40 MHz |
| CPU | 240 MHz dual core Xtensa LX6, cset confirmed by `cpu_start: cpu freq` |
| PSRAM | **none** — no `heap_init` line for SPIRAM and no PSRAM on the module |
| int8 SIMD | **none** — no `esp-dsp` int8 path; Xtensa LX6 has no 8-bit dot-product instruction |

This contradicts the earlier plan's premise of an ESP32-S3 with 8 MB PSRAM. The
difference is decisive: an S3 could hold a model in PSRAM, this part cannot.

### Usable SRAM, measured

ESP-IDF reports the dynamic-allocation regions at boot:

```
I (479) heap_init: At 3FFAE6E0 len 00001920 (6 KiB): DRAM
I (484) heap_init: At 3FFCDAF8 len 00012508 (73 KiB): DRAM
I (491) heap_init: At 3FFE0440 len 00003AE0 (14 KiB): D/IRAM
I (497) heap_init: At 3FFE4350 len 0001BCB0 (111 KiB): D/IRAM
I (503) heap_init: At 4008CAB8 len 00013548 (77 KiB): IRAM
```

6 + 73 + 14 + 111 + 77 = **281 KiB (0.274 MiB) of usable heap**, of which
**190 KiB was still free** after the application and its static arena were
linked in (the firmware logs `heap before load 194788 after 194788`). The brief
estimated "~300 KB usable"; the measured figure is 281 KiB, i.e. slightly
**tighter** than assumed. Nothing here changes the conclusion, and the exact
value is now on record.

## 2. Per-model, per-precision bytes vs the two budgets

Source: `results/quant/sweep_dataset.csv` (t6, dataset.csv, seed 42). `total_mb`
is packed weights + scales + zero points + unquantized parameters + buffers, so
it is the figure that has to fit — not `weights_mb`. Every figure in this report
is dataset.csv; the withdrawn icaisf sweep is not cited anywhere here.

Budgets: **0.274 MiB (281 KiB) usable heap**, **4 MB flash** (~3.9 MB usable
after bootloader, partition table and NVS).

Classification, stated up front so it is not inferred from the table:
`impossible` = exceeds the budget as written; `feasible-flash-mapped` = fits
flash with **≥ 25 % of the 4 MB device left** for application, bootloader and
future growth; `marginal` = fits flash but with less than that headroom.

| model | fp32 | fp16/bf16 | int8 | q8 | q4 | q2 |
|---|---|---|---|---|---|---|
| lejepa | 2.264 | 1.132 | 0.609 | 0.634 | 0.398 | 0.346 |
| t_jepa | 2.534 | 1.267 | 0.674 | 0.701 | 0.436 | 0.377 |
| ts_jepa | 8.613 | 4.307 | 2.231 | 2.361 | 1.459 | 1.258 |
| cf_jepa | 4.929 | 2.465 | 1.287 | 1.358 | 0.843 | 0.728 |
| sigreg | 2.799 | 1.399 | 0.729 | 0.776 | 0.483 | 0.419 |

All values in MiB. RAM residency is a different question with a uniform answer:

| Budget | Verdict for every model, every precision |
|---|---|
| Heap-resident (0.274 MiB) | **impossible for all five at all precisions.** The smallest, lejepa q2 at 0.346 MiB, exceeds the whole usable heap by 0.072 MiB (74 KiB) before a single activation or the C stack is allocated. |
| Flash-mapped, int8 | **feasible-flash-mapped** for lejepa (0.609), t_jepa (0.674), cf_jepa (1.287), sigreg (0.729); **marginal** for ts_jepa (2.231 MiB, leaving ~1.7 MB). |
| Flash-mapped, fp16/bf16 | feasible for lejepa, t_jepa, cf_jepa, sigreg; **impossible for ts_jepa** (4.307 MiB exceeds the entire 4 MB flash). |
| Flash-mapped, q4/q2 | fits by size for all five, but see the accuracy caveat below. |
| fp32 | lejepa/t_jepa/sigreg fit; **cf_jepa impossible** (4.929) and **ts_jepa impossible** (8.613). |

Arithmetic for the two tight rows, shown:
- lejepa q2, RAM: 0.346 MiB = 362,758 B > 281 KiB = 287,744 B. Short by 75,014 B.
- ts_jepa int8, flash: 2.231 MiB = 2,339,468 B; 4 MB = 4,194,304 B → 1,854,836 B
  left, ~1.77 MiB, i.e. 44 % of the device. Fits.
- ts_jepa fp16, flash: 4.307 MiB = 4,516,151 B > 4,194,304 B. Impossible by
  321,847 B.

### The q2 caveat, stated rather than buried

q2 is the smallest option at every model and is the reason the RAM comparison
exists at all, but it is **not a usable operating point**:

| model | fp32 accuracy | int8 accuracy | q2 accuracy | q2 MCC |
|---|---|---|---|---|
| lejepa | 0.935 | 0.933 | 0.489 | 0.323 |
| t_jepa | 0.923 | 0.923 | 0.477 | 0.339 |
| ts_jepa | 0.939 | 0.939 | 0.462 | 0.287 |
| cf_jepa | 0.930 | 0.929 | 0.563 | 0.453 |
| sigreg | 0.615 | 0.615 | 0.458 | 0.281 |

Four-class prediction, so 0.25 is chance. q2 collapses to near chance on every
model while int8 costs **at most 0.002 accuracy** against fp32. So comparing q2
bytes against the heap is the wrong bar: **int8 is simultaneously the only
accurate precision and the smallest fittable one**, and the two arguments point
the same way rather than one rescuing the other.

## 3. Can the paper's JEPA models run on this part?

**Yes, all four, at int8, and only by keeping every weight in flash.** No JEPA
model can be RAM-resident at any precision (section 2), so the memory-mapped
weights are the architecture, not an optimisation.

The two constraints are independent and both are satisfied with margin:

1. **Capacity.** The weight set has to live in 4 MB of flash, not in 281 KiB of
   RAM. At int8 the four JEPA models are 0.609–2.231 MiB, so the three smallest
   leave 1.7–3.3 MB of flash free and ts_jepa leaves 1.7 MB.
2. **Read bandwidth.** Every inference walks the weight set once through the
   flash cache, because nothing is cached between passes. For the demonstrated
   model that is **589,460 bytes per forward pass**, measured on the device. At
   the measured 637 ms per pass (240 MHz) that is about 0.88 MB/s of flash
   traffic — consistent with a cache-line-limited read of a 617 KB blob, and the
   reason the run time is dominated by weight traffic rather than by arithmetic.

The JEPA models therefore do not fail on this part. What they cannot do is run
from RAM, which the earlier ESP32-S3/PSRAM plan assumed they would.

### Demonstration actually run on the device

`firmware/` builds for `esp32` with ESP-IDF v5.3.2, and the lejepa int8 weight
blob is embedded as a binary object so the linker places it in `.rodata` on
flash and the cache maps it directly — the weights are never copied to RAM:

```
I (529) inmotion: blob 617604 bytes at 0x3f4079fc (flash, cache-mapped)
I (539) inmotion: loaded tokens=10 d_model=128 layers=4 classes=4 tail=proj+bn
I (539) inmotion: heap before load 194788 after 194788 (static arena, no per-run alloc)
READY
```

Host protocol (the simple loop the brief asked for): one serial line in, 40
whitespace-separated floats — 10 timesteps x 4 channels, because that is the
input shape dataset.csv and every checkpoint in the sweep use — and one line out:

```
RESULT class=1 confidence=0.869386 latency_ms=887.9 heap_delta=0 heap_free=193872 flash_bytes=589460
```

Measured on device, 48 inferences (16 real test windows x 3 passes):

| CPU clock | median | min | max | heap delta during run | flash bytes/forward |
|---|---|---|---|---|---|
| 160 MHz | 888.5 ms | 879.8 ms | 893.7 ms | 0 B | 589,460 |
| 240 MHz | **636.7 ms** | 630.9 ms | 640.2 ms | 0 B | 589,460 |

Raw per-run data lives in `results/quant/esp32_lejepa_160mhz.json` and
`..._240mhz.json`. That directory is gitignored, so the summary the report cites
is tracked separately at `firmware/evidence/device_measurements.json`, together
with the host-parity result and the chip identification.

Peak RAM as reported by ESP-IDF, not estimated: the forward pass uses a static
arena in `.bss` and allocates nothing, so the measured heap delta across a run is
**0 bytes** and free heap after every run was 193,872 B. The 20 % cut from
160 MHz to 240 MHz (888.5 → 636.7 ms) rather than 33 % is itself evidence the run
is limited by flash/cache traffic as much as by instruction rate.

Correctness on device: the firmware's predicted class agreed with PyTorch's
argmax on **16/16** test windows at both clock settings (`results/quant/esp32_lejepa_*.json`
checked against the golden vectors). The arithmetic itself is proved on the host
first: `firmware/host/jepa_model_test.tsv.c` runs the *same* C forward pass
against PyTorch's logits and reports **max|diff| = 6.884e-06** against a 2e-4
tolerance — the same order as the Rust port's 2.86e-06 for lejepa fp32, and the
reason a device/host disagreement would be a transport bug rather than a
numerical one.

### Why `.rodata` rather than `esp_partition_mmap`

The brief suggested `esp_partition_mmap`. This firmware embeds the blob with
`EMBED_FILES` instead, so the linker places it in `.rodata`, which on the ESP32
is the flash-mapped DROM region: the cache serves reads from SPI flash at the
same address the code sees (`0x3f4079fc` in the log above, inside the
`0x3f400000` DROM window). Both mechanisms end at the same place — weights in
flash, read through the cache, never copied to RAM — and the measured evidence
for that is the identical `heap before load 194788 after 194788`: **loading the
617,604-byte blob consumed zero heap bytes**, which is only possible if nothing
was copied.

The difference is where the bytes sit on the flash: a partition mapping lets the
blob live outside the application image and be updated without reflashing the
app, whereas `.rodata` ties it to the image and needs the 1 MB app partition to
hold both (the built image is 828,656 B, 21 % of the partition free). For a
single fixed model on a demo board the embedded form is the smaller change and
carries no offset arithmetic; if the weights later need to be updated
independently of the firmware, `esp_partition_mmap` over a dedicated partition is
the right move and the loader's single `base` pointer is the only thing that has
to change.

## 4. Open question, not owned here

**sigreg scores 0.487 MCC on dataset.csv against 0.886 on the withdrawn icaisf
sweep, while the four JEPA models move by about +0.03.** Near chance for a
four-class problem. The movement is specific to sigreg and not explained by the
dataset switch, by quantization (fp32 and int8 agree to four decimals, so this
is not a precision effect), or by architecture size (sigreg is larger than
lejepa in parameters). It is recorded here because sigreg appears in the
feasibility table; the investigation belongs to verification (t9), not to this
task, and no explanation is asserted.

## 5. A defect found while doing this

`inmotion-infer predict` rejects a freshly exported lejepa artifact:

```
error: artifact does not record `n_tokens`; re-export it from the registry entry
```

The lejepa artifact's `meta.arch` carries `d_model`, `nhead`, `num_layers`,
`family`, `tokenizer` and `tail` but no `n_tokens`, and the Rust loader requires
it. ts_jepa and cf_jepa export `n_tokens` but lejepa does not, so the existing
`check` golden-vector harness cannot currently be pointed at a lejepa artifact at
all. Reported, not fixed: `rust/` is out of scope for this task. The on-device
parity above does not depend on it — the C path derives `n_tokens` from the
`encoder.pos_embed` tensor's shape and was verified against PyTorch directly.

## 6. Reproduction

```bash
# 1. artifact + golden vectors for the demonstrated model
uv run python -c "from pathlib import Path; from inmotion.quant.export import export, write_golden; \
  export('lejepa', Path('/tmp/lejepa_int8'), quant='int8'); \
  write_golden('lejepa', Path('/tmp/lejepa_int8'), quant='int8')"

# 2. flash blob from that artifact (no re-quantization: it re-lays-out only)
uv run python firmware/tools/export_weights.py \
  --artifact /tmp/lejepa_int8 --out firmware/main/lejepa_q8.bin

# 3. host parity: same C forward pass vs PyTorch logits
uv run python firmware/tools/dump_cases.py /tmp/lejepa_int8/golden.json /tmp/lejepa_int8/cases.tsv
cc -O2 -Ifirmware/main firmware/host/jepa_model_test.tsv.c \
   firmware/main/jepa_model.c firmware/main/jepa_ops.c -o /tmp/jepa_host -lm
/tmp/jepa_host firmware/main/lejepa_q8.bin /tmp/lejepa_int8/cases.tsv
#   -> max|diff| 6.884e-06, 16/16 class agreement, PASS at 2e-4

# 4. build + flash + measure on the device
cd firmware && . $IDF_PATH/export.sh
idf.py set-target esp32 && idf.py build
sg dialout -c 'idf.py -p /dev/ttyUSB0 -b 460800 flash'
sg dialout -c 'cd .. && uv run --extra esp python firmware/tools/run_on_device.py \
  --port /dev/ttyUSB0 --cases /tmp/lejepa_int8/cases.tsv \
  --out results/quant/esp32_lejepa.json --repeats 3'
```

Notes for whoever runs this next: the serial port needs `sg dialout` because the
session's group list is stale, and the DTR/RTS lines must be released before
reading or the port yields line noise instead of boot output (handled in
`run_on_device.py`). Flash was written with `idf.py flash`; the flash was never
erased.

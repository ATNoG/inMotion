"""Pack a lejepa int8 artifact into the flat blob the ESP32 firmware mmaps.

The device must dequantize, not the host: a host-side dequantize would expand
the payload back to 2.26 MB of fp32 and destroy the only reason int8 is on the
table (0.578 MB of flash). So the blob carries folded unsigned codes, per-channel
scales and zero points, and the C kernel does `(q - zp) * scale` inline.

Input is the artifact written by ``inmotion.quant.export``; this script only
re-lays-out data that already exists. It does not re-quantize and does not make
a second source of truth for any number.

Usage:
    uv run python firmware/tools/export_weights.py --artifact /tmp/lejepa_int8 --out firmware/main/lejepa_q8.bin
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from safetensors.torch import load_file

MAGIC = b"INMJQ8\x00\x00"
# name[64] + shape[4]*u32 + n_dims u32 + kind u32 + codes_off u32
# + scale_off u32 + zp_off u32 + n_codes u32
ENTRY = struct.Struct("<64s4I6I")


class Tensor:
    """One blob entry, either int8-coded or fp32."""

    def __init__(self, name: str, shape: list[int], kind: str, payload: bytes) -> None:
        self.name = name
        self.shape = shape
        self.kind = kind
        self.payload = payload


def build(artifact: Path) -> tuple[bytes, dict[str, object]]:
    tensors = load_file(str(artifact / "model.safetensors"))
    meta = json.loads((artifact / "meta.json").read_text())
    index: dict[str, dict[str, object]] = meta["tensors"]

    entries: list[Tensor] = []
    for name in sorted(index):
        info = index[name]
        shape = [int(s) for s in info["shape"]]
        if info["quantized"]:
            codes = tensors[f"{name}.q"].numpy().tobytes()
            scale = tensors[f"{name}.scale"].numpy().astype("<f4").tobytes()
            # Codes and zero point are both biased by 1 << (bits-1): pack() adds
            # it to the codes and fold_bias_into_zero_point adds it to the zero
            # point. The two must stay biased together so (byte - zp) recovers
            # (code - zero_point). Un-biasing only the zero point would shift
            # every weight by 128*scale.
            # Stored as one unsigned byte: the packed zero point lives in
            # 0..255 for 8-bit codes, so a signed byte would wrap at 128.
            zp = tensors[f"{name}.zp"].numpy().astype("<u1").tobytes()
            entries.append(Tensor(name, shape, "i8", codes))
            # Payload is laid out codes | scale | zp so one entry holds all three.
            # Offsets are patched below; keep scale/zp alongside.
            entries[-1].scale = scale
            entries[-1].zp = zp
        else:
            entries.append(
                Tensor(name, shape, "f32", tensors[name].numpy().astype("<f4").tobytes())
            )

    # Layout: header | entry table | payloads (16-byte aligned)
    header = struct.Struct("<8sIIIII")
    # entry_size is written rather than assumed: the reader walks the table with
    # it, so a stride change cannot silently misread every tensor.
    assert ENTRY.size == 104, ENTRY.size
    table_offset = header.size
    payload_offset = table_offset + ENTRY.size * len(entries)
    payload_offset = (payload_offset + 15) & ~15

    table = bytearray()
    payload = bytearray()
    cursor = payload_offset
    for t in entries:
        if t.kind == "i8":
            codes_off = cursor
            payload += t.payload
            cursor += len(t.payload)
            scale_off = cursor
            payload += t.scale
            cursor += len(t.scale)
            zp_off = cursor
            payload += t.zp
            cursor += len(t.zp)
            n_codes = len(t.payload)
        else:
            codes_off, scale_off, zp_off, n_codes = cursor, 0, 0, 0
            payload += t.payload
            cursor += len(t.payload)
            payload += b"\x00" * (-cursor % 16)
            cursor = payload_offset + len(payload)
        padded = t.name.encode()[:63]
        shape = [int(s) for s in t.shape] + [1, 1, 1, 1]
        kind = 0 if t.kind == "f32" else 1
        table += ENTRY.pack(
            padded, shape[0], shape[1], shape[2], shape[3],
            len(t.shape),          # n_dims
            kind,
            codes_off, scale_off, zp_off, n_codes,
        )

    blob = bytearray()
    blob += header.pack(MAGIC, 1, len(entries), table_offset, payload_offset, ENTRY.size)
    blob += table
    blob += b"\x00" * (payload_offset - len(blob))
    blob += payload
    return bytes(blob), {"entries": len(entries), "bytes": len(blob), "payload_offset": payload_offset}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    blob, info = build(args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(f"wrote {args.out} entries={info['entries']} bytes={info['bytes']}")


if __name__ == "__main__":
    main()

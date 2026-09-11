"""Bit-packing of integer codes.

Storing a 4-bit code in an int8 wastes half the memory, and this workload is
memory-bound, so the packing is not an optimization — it is the point. Q4 must
cost half of Q8 and Q2 a quarter of it, or the exercise is pointless.

Format
------
Codes are biased to unsigned (``u = q + 2**(bits-1)``) and written LSB-first:
element ``i`` occupies bits ``[i*bits, (i+1)*bits)`` of the byte stream, so for
4-bit codes element 0 is the low nibble of byte 0 and element 1 the high nibble.

Biasing means the on-device dequantization stays a single formula with no extra
per-weight work: because ``q = u - 2**(bits-1)``,

    (q - zero_point) * scale  ==  (u - (zero_point + 2**(bits-1))) * scale

so the bias is folded into the zero point once, at export, by
:func:`fold_bias_into_zero_point`.
"""

from __future__ import annotations

import numpy as np

# Bit widths with a vectorized packing path. Others fall back to a scalar loop.
_VECTORIZED = (1, 2, 4, 8)


def n_bytes_for(count: int, bits: int) -> int:
    """Number of bytes needed to pack ``count`` codes of ``bits`` each."""
    return (count * bits + 7) // 8


def fold_bias_into_zero_point(zero_point: np.ndarray, bits: int) -> np.ndarray:
    """Return the zero point that pairs with biased unsigned codes."""
    bias = 1 << (bits - 1)
    return (zero_point.astype(np.int32) + bias).astype(np.int32)


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack int8 codes into a uint8 byte stream, LSB-first."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    flat = np.asarray(codes, dtype=np.int64).ravel()
    biased = (flat + (1 << (bits - 1))).astype(np.int64)
    if biased.min() < 0 or biased.max() >= (1 << bits):
        raise ValueError("code out of range for the given bit width")

    if bits == 8:
        return biased.astype(np.uint8)

    count = flat.size
    out = np.zeros(n_bytes_for(count, bits), dtype=np.uint8)

    if bits in (1, 2, 4):
        per_byte = 8 // bits
        n_full = count // per_byte
        if n_full:
            trimmed = biased[: n_full * per_byte].reshape(n_full, per_byte)
            acc = np.zeros(n_full, dtype=np.uint16)
            for slot in range(per_byte):
                acc |= trimmed[:, slot].astype(np.uint16) << (slot * bits)
            out[:n_full] = acc.astype(np.uint8)
        tail = biased[n_full * per_byte :]
        acc = 0
        for slot, value in enumerate(tail):
            acc |= int(value) << (slot * bits)
        if tail.size:
            out[n_full] = np.uint8(acc)
        return out

    # General widths (3, 5, 6, 7): scalar, but only used off-device.
    for i, value in enumerate(biased):
        bitpos = i * bits
        byte, off = bitpos >> 3, bitpos & 7
        v = int(value)
        out[byte] |= np.uint8((v << off) & 0xFF)
        if off + bits > 8:
            out[byte + 1] |= np.uint8((v >> (8 - off)) & 0xFF)
    return out


def unpack(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Inverse of :func:`pack`. Returns signed int8 codes."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    data = np.asarray(packed, dtype=np.uint8).ravel()
    bias = 1 << (bits - 1)

    if bits == 8:
        codes = data[:count].astype(np.int64)
    elif bits in (1, 2, 4):
        per_byte = 8 // bits
        mask = (1 << bits) - 1
        n_full = count // per_byte
        parts = []
        if n_full:
            chunk = data[:n_full].astype(np.int64)
            # Element i*per_byte + slot lives at bit slot*bits of byte i, so the
            # per-byte slots must be interleaved, not concatenated.
            cols = np.stack([(chunk >> (slot * bits)) & mask for slot in range(per_byte)], axis=1)
            parts.append(cols.reshape(-1))
        tail_count = count - n_full * per_byte
        if tail_count:
            last = int(data[n_full]) if n_full < data.size else 0
            parts.append(
                np.array(
                    [(last >> (slot * bits)) & mask for slot in range(tail_count)],
                    dtype=np.int64,
                )
            )
        codes = np.concatenate(parts) if len(parts) > 1 else parts[0]
    else:
        codes = np.zeros(count, dtype=np.int64)
        for i in range(count):
            bitpos = i * bits
            byte, off = bitpos >> 3, bitpos & 7
            v = int(data[byte]) >> off
            if off + bits > 8:
                v |= int(data[byte + 1]) << (8 - off)
            codes[i] = v & ((1 << bits) - 1)

    return (codes - bias).astype(np.int8)

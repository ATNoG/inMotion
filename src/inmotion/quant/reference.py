"""Normative quantization math, in numpy.

This module *is* the specification. The Rust kernels in ``rust/inmotion-core``
must reproduce it exactly, and ``tests/test_quant_reference.py`` pins the
properties that make "exactly" checkable:

1. dequantize is exactly ``(q - zero_point) * scale`` in float32;
2. re-quantizing a dequantized tensor is idempotent (no compounding drift);
3. rounding ties follow the declared mode, because numpy rounds half to even
   while Rust rounds half away from zero;
4. granularity is honoured, so per-group is never silently coarser than asked;
5. the byte accounting uses the packed size, not one int8 per value.

Layout convention
-----------------
``scale`` and ``zero_point`` are stored in the *reduced* shape that broadcasts
against the codes, so no reader has to re-derive it:

===============  ==========================================  ==================
granularity      reduced shape                               dequant broadcast
===============  ==========================================  ==================
per_tensor       ``()``                                      scalar
per_channel      ``(1, .., D, .., 1)`` with ``D=shape[axis]``  direct
per_group        ``(*shape[:-1], ceil(D/g))``                ``scale[..., None]``
===============  ==========================================  ==================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inmotion.quant.spec import MIN_SCALE, Granularity, QuantSpec, Rounding

F32 = np.float32


def _round(values: np.ndarray, mode: Rounding) -> np.ndarray:
    """Round to nearest integer with an explicit tie rule."""
    if mode is Rounding.HALF_TO_EVEN:
        # numpy.round implements IEEE roundTiesToEven.
        return np.round(values)
    if mode is Rounding.HALF_AWAY:
        # sign * floor(|x| + 0.5): -0.5 -> -1, +0.5 -> +1
        return np.sign(values) * np.floor(np.abs(values) + 0.5)
    raise ValueError(f"unknown rounding mode {mode!r}")


def n_groups_for(length: int, group_size: int) -> int:
    """Number of groups the last axis is divided into, rounding up."""
    return -(-length // group_size)


@dataclass
class QuantizedTensor:
    """Integer codes plus the metadata needed to reconstruct the float values."""

    q: np.ndarray
    """Integer codes, same shape as the original tensor.

    ``int8`` for a symmetric scheme, ``uint8`` for an asymmetric one, whose
    codes run ``0..2**bits - 1``.
    """
    scale: np.ndarray
    """float32 scale, in the reduced shape documented above."""
    zero_point: np.ndarray
    """Zero point, same reduced shape as ``scale``, and the same dtype as ``q``."""
    shape: tuple[int, ...]
    """Shape of the original tensor."""
    spec: QuantSpec

    @property
    def n_values(self) -> int:
        return int(self.q.size)

    @property
    def n_bytes(self) -> int:
        """Packed payload plus float32 scales plus int8 zero points."""
        return (
            (int(self.q.size) * self.spec.bits + 7) // 8
            + 4 * int(self.scale.size)
            + int(self.zero_point.size)
        )


def _shape_of_reduction(shape: tuple[int, ...], spec: QuantSpec) -> tuple[int, ...]:
    """Reduced scale/zero-point shape, used by readers to validate metadata."""
    if spec.granularity is Granularity.PER_TENSOR:
        return ()
    if spec.granularity is Granularity.PER_CHANNEL:
        axis = spec.axis % len(shape)
        return tuple(1 if i != axis else shape[i] for i in range(len(shape)))
    return (
        *shape[:-1],
        n_groups_for(shape[-1], spec.group_size),
    )


def quantize(x: np.ndarray, spec: QuantSpec) -> QuantizedTensor:
    """Quantize a float tensor into integer codes and per-group metadata."""
    xf = np.asarray(x, dtype=F32)
    if xf.ndim == 0:
        xf = xf.reshape(1)
    shape = tuple(int(d) for d in xf.shape)
    qmin, qmax = spec.qmin, spec.qmax

    # Build a view whose last axis is the reduction axis, and the axes to reduce
    # over. Per-channel reduces every axis except `axis`; per-group reduces only
    # the trailing group axis.
    if spec.granularity is Granularity.PER_GROUP:
        n_groups = n_groups_for(shape[-1], spec.group_size)
        pad = n_groups * spec.group_size - shape[-1]
        if pad:
            xf = np.pad(xf, [(0, 0)] * (xf.ndim - 1) + [(0, pad)])
        view = xf.reshape(*shape[:-1], n_groups, spec.group_size)
        reduce_axes: tuple[int, ...] | int | None = -1
        keepdims = True
    elif spec.granularity is Granularity.PER_CHANNEL:
        axis = spec.axis % len(shape)
        view = xf
        reduce_axes = tuple(i for i in range(xf.ndim) if i != axis)
        keepdims = True
    else:
        view = xf
        reduce_axes = tuple(range(xf.ndim))
        keepdims = False

    def reduce_max(values: np.ndarray) -> np.ndarray:
        if reduce_axes is None:
            return np.max(values)
        return np.max(values, axis=reduce_axes, keepdims=keepdims)

    def reduce_min(values: np.ndarray) -> np.ndarray:
        if reduce_axes is None:
            return np.min(values)
        return np.min(values, axis=reduce_axes, keepdims=keepdims)

    # The dtype follows the scheme. Asymmetric codes are unsigned and reach
    # `2**bits - 1`, which int8 cannot hold: at 8 bits, 255 wraps to -1 and the
    # reconstruction becomes noise rather than a slightly lossy tensor. Symmetric
    # codes are signed and fit int8. The shape is the same either way, so nothing
    # downstream changes.
    code_dtype = np.int8 if spec.symmetric else np.uint8

    if spec.symmetric:
        scale = np.maximum(reduce_max(np.abs(view)) / F32(qmax), F32(MIN_SCALE)).astype(F32)
        zero_point = np.zeros(np.shape(scale), dtype=np.int8)
        codes = _round(view / scale, spec.rounding)
    else:
        vmax, vmin = reduce_max(view), reduce_min(view)
        span = np.maximum(vmax - vmin, F32(MIN_SCALE)).astype(F32)
        scale = (span / F32(qmax - qmin)).astype(F32)
        zp_raw = _round(F32(qmin) - vmin / scale, spec.rounding)
        zero_point = np.clip(zp_raw, qmin, qmax).astype(code_dtype)
        codes = _round(view / scale + zero_point.astype(F32), spec.rounding)

    codes = np.clip(codes, qmin, qmax).astype(code_dtype)

    if spec.granularity is Granularity.PER_GROUP:
        # The reduction kept the group axis as a singleton; the documented
        # reduced shape is (*shape[:-1], n_groups), so drop it here. Readers
        # then broadcast with scale[..., None].
        scale = scale[..., 0]
        zero_point = zero_point[..., 0]
        # Discard the zero padding introduced above.
        codes = codes.reshape(*shape[:-1], n_groups * spec.group_size)[..., : shape[-1]]

    return QuantizedTensor(
        q=codes.reshape(shape),
        scale=scale.astype(F32),
        zero_point=zero_point.astype(np.int8),
        shape=shape,
        spec=spec,
    )


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Reconstruct float32 values from codes and metadata.

    The device performs exactly this arithmetic: widen the code, subtract the
    zero point, multiply by the scale. Per-group codes are re-viewed as
    ``(..., n_groups, group_size)`` and the padding is dropped.
    """
    spec = qt.spec
    q = qt.q.astype(F32)
    zp = qt.zero_point.astype(F32)

    if spec.granularity is Granularity.PER_GROUP:
        n_groups = n_groups_for(qt.shape[-1], spec.group_size)
        total = n_groups * spec.group_size
        # Re-introduce the same zero padding `quantize` used, so the codes can be
        # viewed group-wise; the padding is dropped again on the way out.
        padded = q
        if total != qt.shape[-1]:
            padded = np.pad(q, [(0, 0)] * (q.ndim - 1) + [(0, total - qt.shape[-1])])
        grouped = padded.reshape(*qt.shape[:-1], n_groups, spec.group_size)
        out = (grouped - zp[..., None]) * qt.scale[..., None]
        return out.reshape(*qt.shape[:-1], total)[..., : qt.shape[-1]].astype(F32)

    # per_tensor: scalar metadata. per_channel: metadata already carries the
    # singleton axes, so it broadcasts directly.
    return ((q - zp) * qt.scale).astype(F32)

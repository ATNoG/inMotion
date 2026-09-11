"""Quantization schemes.

Weight-only quantization: the weights are stored at low precision, activations
stay in fp32. That is the right trade here because weights dominate this
workload by 9x-2424x over activations (a full inference window is 10 timesteps
x 4 channels = 160 bytes), so the entire memory problem is the weights. Keeping
activations in fp32 also removes the need for calibration data, which keeps Q4
and Q8 exact and reproducible rather than data-dependent.

Rounding is specified explicitly. NumPy's ``np.round`` rounds half to even,
while Rust's ``f32::round`` rounds half away from zero; leaving that implicit
would make the Python reference and the Rust kernels disagree on roughly one
element in a thousand, which is exactly the kind of drift the golden-vector
tests exist to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Smallest magnitude that may be used as a scale. Guards against all-zero
# groups and keeps the reciprocal finite.
MIN_SCALE = 1e-8


class Granularity(str, Enum):
    """How many weights share one scale/zero-point pair."""

    PER_TENSOR = "per_tensor"
    """One pair for the whole tensor. Smallest metadata, largest error."""
    PER_CHANNEL = "per_channel"
    """One pair per slice along ``axis`` (per output row for nn.Linear)."""
    PER_GROUP = "per_group"
    """One pair per contiguous block of ``group_size`` elements."""


class Rounding(str, Enum):
    """Rounding mode applied before the integer cast."""

    HALF_TO_EVEN = "half_to_even"
    """Round half to even, matching numpy.round and IEEE 754 roundTiesToEven."""
    HALF_AWAY = "half_away"
    """Round half away from zero, matching Rust's f32::round."""


@dataclass(frozen=True)
class QuantSpec:
    """A weight-quantization scheme.

    ``bits`` is the storage precision of the *values*; the array dtype is always
    int8 so the payload has one shape regardless of bit width, which keeps the
    reader simple on the device side.
    """

    bits: int = 8
    granularity: Granularity = Granularity.PER_CHANNEL
    symmetric: bool = True
    group_size: int = 32
    axis: int = 0
    rounding: Rounding = Rounding.HALF_TO_EVEN
    name: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 8:
            raise ValueError(f"bits must be in 1..8, got {self.bits}")
        if self.granularity is Granularity.PER_GROUP and self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def qmin(self) -> int:
        """Smallest representable integer code."""
        return -self.qmax - 1 if self.symmetric else 0

    @property
    def qmax(self) -> int:
        """Largest representable integer code."""
        return (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1

    @property
    def label(self) -> str:
        """Stable short name, used in filenames and result tables."""
        if self.name:
            return self.name
        sym = "sym" if self.symmetric else "asym"
        if self.granularity is Granularity.PER_GROUP:
            gran = f"g{self.group_size}"
        elif self.granularity is Granularity.PER_CHANNEL:
            gran = "ch"
        else:
            gran = "tensor"
        return f"q{self.bits}_{gran}_{sym}"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.label,
            "bits": self.bits,
            "granularity": self.granularity.value,
            "symmetric": self.symmetric,
            "group_size": self.group_size,
            "axis": self.axis,
            "rounding": self.rounding.value,
        }


# Schemes offered on the command line. `fp32`/`fp16`/`bf16` are dtype casts, not
# affine quantization, and are handled by the exporter rather than the reference
# quantizer.
PRESET_SPECS: dict[str, QuantSpec] = {
    "int8": QuantSpec(bits=8, granularity=Granularity.PER_CHANNEL, symmetric=True),
    "q8": QuantSpec(bits=8, granularity=Granularity.PER_GROUP, group_size=64, symmetric=True),
    "q6": QuantSpec(bits=6, granularity=Granularity.PER_GROUP, group_size=64, symmetric=True),
    "q5": QuantSpec(bits=5, granularity=Granularity.PER_GROUP, group_size=64, symmetric=True),
    "q4": QuantSpec(bits=4, granularity=Granularity.PER_GROUP, group_size=32, symmetric=True),
    "q3": QuantSpec(bits=3, granularity=Granularity.PER_GROUP, group_size=32, symmetric=True),
    "q2": QuantSpec(bits=2, granularity=Granularity.PER_GROUP, group_size=16, symmetric=True),
}

# Cast-only precisions, produced by the exporter.
FLOAT_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16")


def resolve(label: str) -> QuantSpec:
    """Look up a preset by name, or parse ``q<N>`` shorthand."""
    if label in PRESET_SPECS:
        return PRESET_SPECS[label]
    raise KeyError(
        f"unknown quantization {label!r}; "
        f"known: {', '.join(sorted(PRESET_SPECS))} or {', '.join(FLOAT_PRECISIONS)}"
    )

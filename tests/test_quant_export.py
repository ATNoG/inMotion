"""Contract for the export artifact and the Rust parity check.

The artifact is the interface between two languages, so its shape is a contract:
if a tensor is missing, or a buffer is omitted, the Rust side fails to load and
the failure surfaces far from the cause. `encoder.bn.running_mean` was exactly
that bug — it is a *buffer*, not a parameter, and an exporter that walked only
`named_parameters` produced an artifact the runtime could not load.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import REPO_ROOT, requires_artifacts

from inmotion.quant.export import WEIGHTS_NAME, export, write_golden

MODEL = "lejepa"
RUST_DIR = REPO_ROOT / "rust"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    if not (REPO_ROOT / "backup").is_dir():
        pytest.skip("checkpoint store not present")
    out = tmp_path_factory.mktemp("artifact") / MODEL
    export(MODEL, out, quant=None)
    write_golden(MODEL, out, quant=None)
    return out


@requires_artifacts
def test_export_writes_the_expected_files(artifact: Path) -> None:
    assert (artifact / WEIGHTS_NAME).is_file()
    assert (artifact / "meta.json").is_file()
    assert (artifact / "golden.json").is_file()


@requires_artifacts
def test_meta_describes_every_tensor(artifact: Path) -> None:
    meta = json.loads((artifact / "meta.json").read_text())
    assert meta["artifact_version"] == 1
    assert meta["model"] == MODEL
    assert meta["seq_len"] == 10
    assert meta["in_channels"] == 4
    assert meta["num_classes"] == 4
    assert meta["eps"] == {"layernorm": 1e-5, "batchnorm": 1e-5}
    assert meta["tensors"], "no tensors described"


@requires_artifacts
@pytest.mark.parametrize(
    "name",
    [
        "encoder.tokenizer.weight",
        "encoder.pos_embed",
        "encoder.bn.running_mean",
        "encoder.bn.running_var",
        "attn.weight",
        "head.3.weight",
    ],
)
def test_required_tensors_are_present(artifact: Path, name: str) -> None:
    """Buffers count: `running_mean`/`running_var` are needed at inference."""
    meta = json.loads((artifact / "meta.json").read_text())
    assert name in meta["tensors"], f"{name} missing from the artifact"


@requires_artifacts
def test_qkv_is_split_so_the_runtime_needs_no_packing_convention(artifact: Path) -> None:
    """PyTorch packs QKV into one matrix; the exporter splits it."""
    meta = json.loads((artifact / "meta.json").read_text())["tensors"]
    for tag in ("q", "k", "v"):
        assert f"encoder.transformer.layers.0.self_attn.{tag}.weight" in meta
    assert "encoder.transformer.layers.0.self_attn.in_proj_weight" not in meta


@requires_artifacts
@pytest.mark.parametrize("quant", ["int8", "q4"])
def test_quantized_export_carries_packed_metadata(
    artifact: Path, quant: str, tmp_path: Path
) -> None:
    out = tmp_path / quant
    export(MODEL, out, quant=quant)
    meta = json.loads((out / "meta.json").read_text())
    quantized = {n: i for n, i in meta["tensors"].items() if i["quantized"]}
    assert quantized, "nothing was quantized"
    for info in quantized.values():
        assert info["bits"] in (8, 4)
        assert info["shape"]
        assert info["scale_shape"]
    assert meta["quant"]["bits"] in (8, 4)


@requires_artifacts
def test_golden_uses_the_quantized_weights(artifact: Path, tmp_path: Path) -> None:
    """Golden outputs must come from the same weights the artifact holds.

    Comparing a quantized artifact against unquantized outputs measures
    quantization error and reports it as a parity failure.
    """
    fp32 = tmp_path / "fp32"
    q4 = tmp_path / "q4"
    export(MODEL, fp32, quant=None)
    write_golden(MODEL, fp32, quant=None)
    export(MODEL, q4, quant="q4")
    write_golden(MODEL, q4, quant="q4")

    a = json.loads((fp32 / "golden.json").read_text())
    b = json.loads((q4 / "golden.json").read_text())
    assert a["quant"] is None
    assert b["quant"] == "q4"
    first_a, first_b = a["cases"][0]["logits"], b["cases"][0]["logits"]
    assert first_a != first_b, "golden for q4 is identical to fp32; it was not quantized"


def _cargo() -> str:
    return shutil.which("cargo") or ""


@pytest.mark.skipif(not _cargo(), reason="cargo not installed")
@requires_artifacts
@pytest.mark.parametrize("quant", [None, "int8", "q4"])
def test_rust_port_reproduces_pytorch(quant: str | None, tmp_path: Path) -> None:
    """The cross-language gate: export, then have Rust replay the golden vectors.

    A port that is not checked against the original is worthless, so this is the
    test that makes the Rust implementation admissible.
    """
    out = tmp_path / (quant or "fp32")
    export(MODEL, out, quant=quant)
    write_golden(MODEL, out, quant=quant)

    result = subprocess.run(
        [
            _cargo(),
            "run",
            "--release",
            "--quiet",
            "-p",
            "inmotion-cli",
            "--",
            "check",
            "--artifact",
            str(out),
            "--golden",
            str(out),
        ],
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "PARITY OK" in result.stdout, result.stdout

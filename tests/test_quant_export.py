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

#: The models that share one Rust transformer implementation. They differ only
#: in their tokenizer, in T-JEPA's extra projection, and in their tail, so all of
#: them are replayed.
TRANSFORMER_MODELS = ("lejepa", "ts_jepa", "cf_jepa", "t_jepa")

#: Every model the Rust runtime implements. `sigreg` is the second family: a CNN,
#: dispatched from `arch.family` rather than through the transformer path.
RUST_MODELS = (*TRANSFORMER_MODELS, "sigreg")

RUST_DIR = REPO_ROOT / "rust"


@pytest.fixture(scope="session")
def artifact_for(tmp_path_factory):
    """Export ``(model, quant)`` once per session, golden vectors included.

    Exporting is the slow half of these tests and several of them want the same
    artifacts, so they are built once here rather than once per test.
    """
    if not (REPO_ROOT / "backup").is_dir():
        pytest.skip("checkpoint store not present")
    cache: dict[tuple[str, str | None], Path] = {}

    def get(model: str, quant: str | None) -> Path:
        if (model, quant) not in cache:
            out = tmp_path_factory.mktemp(f"{model}-{quant or 'fp32'}") / model
            export(model, out, quant=quant)
            write_golden(model, out, quant=quant)
            cache[(model, quant)] = out
        return cache[(model, quant)]

    return get


@pytest.fixture(scope="module")
def artifact(artifact_for) -> Path:
    return artifact_for(MODEL, None)


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
def test_golden_uses_the_quantized_weights(artifact_for) -> None:
    """Golden outputs must come from the same weights the artifact holds.

    Comparing a quantized artifact against unquantized outputs measures
    quantization error and reports it as a parity failure.
    """
    a = json.loads((artifact_for(MODEL, None) / "golden.json").read_text())
    b = json.loads((artifact_for(MODEL, "q4") / "golden.json").read_text())
    assert a["quant"] is None
    assert b["quant"] == "q4"
    first_a, first_b = a["cases"][0]["logits"], b["cases"][0]["logits"]
    assert first_a != first_b, "golden for q4 is identical to fp32; it was not quantized"


@requires_artifacts
@pytest.mark.parametrize("model", TRANSFORMER_MODELS)
def test_artifact_records_the_architecture_the_runtime_selects(model: str, artifact_for) -> None:
    """The runtime picks its tokenizer and tail from the artifact, not from a name.

    `arch` is read from the module being exported, so it is populated even when
    the registry entry takes its factory defaults (`ts_jepa`, `cf_jepa`).
    """
    out = artifact_for(model, None)
    meta = json.loads((out / "meta.json").read_text())
    arch, tensors = meta["arch"], meta["tensors"]
    assert arch["family"] == "transformer"
    assert arch["tail"] in ("layernorm", "proj_bn")
    assert arch["num_layers"] >= 1
    assert arch["nhead"] >= 1
    assert arch["d_model"] >= 1
    if arch["tokenizer"] == "feature":
        total = arch["total_features"]
        assert arch["n_tokens"] == arch["n_reg"] + total
        assert tensors["encoder.tokenizer.weight"]["shape"] == [arch["n_tokens"], arch["d_model"]]
        for name in ("encoder.tokenizer.bias", "encoder.tokenizer.feat_idx_embed.weight"):
            assert tensors[name]["shape"] == [total, arch["d_model"]]
        # T-JEPA has a feature-index embedding instead of a positional one.
        assert "encoder.pos_embed" not in tensors
    elif arch["tokenizer"] == "patch":
        conv = tensors["encoder.tokenizer.proj.weight"]["shape"]
        assert len(conv) == 3, "a conv weight is 3-D, not a flattened matrix"
        assert conv[0] == arch["d_model"]
        assert conv[2] == arch["patch_size"]
    else:
        assert arch["tokenizer"] == "linear"
        assert "encoder.tokenizer.weight" in tensors


@requires_artifacts
def test_t_jepa_exports_its_per_slot_tokenizer(artifact_for) -> None:
    """T-JEPA's tokenizer is shaped and selected exactly as the reference uses it.

    Two [REG] slots are prepended to the 10 x 18 flattened window, each slot has
    its own weight row, and the bias / feature-index rows cover the features
    only — `FeatureTokenizer.forward` pads them with zeros for the [REG] slots.
    """
    meta = json.loads((artifact_for("t_jepa", None) / "meta.json").read_text())
    arch, tensors = meta["arch"], meta["tensors"]
    assert arch["tokenizer"] == "feature"
    assert arch["input_proj"] is True
    assert arch["n_reg"] == 2
    assert arch["total_features"] == 10 * 18 == 180
    assert arch["n_tokens"] == 182
    assert tensors["encoder.tokenizer.weight"]["shape"] == [182, 128]
    assert tensors["encoder.tokenizer.bias"]["shape"] == [180, 128]
    assert tensors["encoder.tokenizer.feat_idx_embed.weight"]["shape"] == [180, 128]
    assert "encoder.input_proj.weight" in tensors
    # The [REG] input value is the constant 1.0, so the learnable token rows are
    # dead weight in the classifier; the Rust port must not read them.
    assert "encoder.tokenizer.reg_tokens.tokens" in tensors


@requires_artifacts
def test_sigreg_exports_its_cnn_without_the_wrapper_prefix(artifact_for) -> None:
    """SIGReg is a CNN, and `TupleWrap`'s `m.` prefix is resolved at export.

    The registry declares the prefix, so the exporter strips it generically; no
    tensor is renamed by hand and the runtime sees the model's own names.
    """
    meta = json.loads((artifact_for("sigreg", None) / "meta.json").read_text())
    arch, tensors = meta["arch"], meta["tensors"]
    assert arch["family"] == "sigreg_cnn"
    assert arch["num_filters"] == 192
    assert arch["num_blocks"] == 3
    assert arch["latent_dim"] == 128
    assert arch["front_kernels"] == [3, 5, 7]
    assert arch["block_kernels"] == [3, 3]
    assert meta["in_channels"] == 4
    assert not [name for name in tensors if name.startswith("m.")], "wrapper prefix leaked"
    for name in (
        "encoder.conv3.weight",
        "encoder.conv5.weight",
        "encoder.conv7.weight",
        "encoder.bn.running_mean",
        "encoder.blocks.0.block.0.weight",
        "encoder.blocks.0.block.4.running_var",
        "latent_proj.weight",
        "latent_norm.weight",
        "classifier.weight",
    ):
        assert name in tensors, f"{name} missing from the artifact"


@requires_artifacts
@pytest.mark.parametrize("model", ["ts_jepa", "cf_jepa", "sigreg"])
@pytest.mark.parametrize("quant", ["int8", "q4"])
def test_quantized_conv_weight_keeps_its_rank(model: str, quant: str, artifact_for) -> None:
    """`shape` is the original tensor shape, not the quantization view.

    Per-group quantization flattens a Conv1d `(out, in, k)` weight to
    `(out, in*k)` for grouping. Recording that view made the Rust loader hand a
    matrix to `conv1d`, which requires `(out, in, k)`.
    """
    tensors = json.loads((artifact_for(model, quant) / "meta.json").read_text())["tensors"]
    convs = [n for n, i in tensors.items() if i["quantized"] and len(i["shape"]) == 3]
    assert convs, f"{model}: no quantized convolution weight in the artifact"


def _cargo() -> str:
    return shutil.which("cargo") or ""


@pytest.mark.skipif(not _cargo(), reason="cargo not installed")
@requires_artifacts
@pytest.mark.parametrize("model", RUST_MODELS)
@pytest.mark.parametrize("quant", [None, "int8", "q4"])
def test_rust_port_reproduces_pytorch(model: str, quant: str | None, artifact_for) -> None:
    """The cross-language gate: export, then have Rust replay the golden vectors.

    A port that is not checked against the original is worthless, so this is the
    test that makes the Rust implementation admissible for every family: the four
    transformers and the SIGReg CNN.
    """
    out = artifact_for(model, quant)

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

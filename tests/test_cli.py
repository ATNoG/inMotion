"""CLI smoke tests.

These exercise the command surface the way a user would: through the typer app,
asserting on exit codes and parseable output. They deliberately do not poke at
internal functions, so the CLI can be reorganised without rewriting them.
"""

from __future__ import annotations

import json

from conftest import requires_artifacts
from typer.testing import CliRunner

from inmotion.cli import app

runner = CliRunner()


def _invoke(*args: str):
    return runner.invoke(app, list(args))


def test_root_help_lists_command_groups() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    for group in ("model", "datasets", "data", "run", "doctor"):
        assert group in result.stdout


def test_version() -> None:
    result = _invoke("--version")
    assert result.exit_code == 0
    assert "inmotion" in result.stdout


def test_doctor_reports_environment() -> None:
    result = _invoke("doctor")
    assert result.exit_code == 0
    assert "python" in result.stdout
    assert "models" in result.stdout


def test_datasets_list_json_is_machine_readable() -> None:
    result = _invoke("datasets", "list", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["datasets"], "manifest lists no datasets"
    assert payload["canonical"] in {d["id"] for d in payload["datasets"]}


def test_datasets_verify_matches_manifest() -> None:
    """The committed manifest must describe the committed datasets."""
    result = _invoke("datasets", "verify")
    assert result.exit_code == 0, result.stdout


def test_datasets_show_renders_lineage() -> None:
    result = _invoke("datasets", "show", "ds-augmented-v3")
    assert result.exit_code == 0
    assert "lineage" in result.stdout
    assert "label audit" in result.stdout


def test_datasets_resolve_warns_on_known_bad_labels() -> None:
    """A superseded dataset with bad labels must not be handed over silently."""
    result = _invoke("datasets", "resolve", "ds-pure")
    assert "incorrect class labels" in result.stderr or "incorrect class labels" in result.stdout


def test_model_list_json_covers_registry() -> None:
    result = _invoke("model", "list", "--json")
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert rows, "registry is empty"
    assert all({"name", "factory", "params"} <= set(r) for r in rows)


def test_run_list_documents_pipelines() -> None:
    result = _invoke("run", "list")
    assert result.exit_code == 0
    assert "exotic" in result.stdout


def test_unknown_pipeline_exits_nonzero() -> None:
    result = _invoke("run", "pipeline", "not-a-pipeline")
    assert result.exit_code != 0


@requires_artifacts
def test_model_verify_passes_for_lean_model() -> None:
    """One strict load through the CLI, on the cheapest checkpoint."""
    result = _invoke("model", "verify", "sigreg_after_hpo")
    assert result.exit_code == 0, result.stdout


@requires_artifacts
def test_model_info_surfaces_provenance_gap() -> None:
    result = _invoke("model", "info", "mamba3_tcn")
    assert result.exit_code == 0
    assert "DIFFERS" in result.stdout
    assert "provenance gap" in result.stdout


@requires_artifacts
def test_model_adopt_reports_ambiguity() -> None:
    """TS-JEPA and CF-JEPA at equal width are indistinguishable from weights."""
    result = _invoke(
        "model",
        "adopt",
        "20-aug/models/exotic/normal-new-ds/cf_jepa_ft_seed42.pt",
        "--name",
        "cf_jepa_base",
    )
    assert result.exit_code == 0, result.stdout
    assert "cf_jepa" in result.stdout
    assert "ambiguous with" in result.stdout

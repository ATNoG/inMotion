"""Generate the reference documentation from the code itself.

Run from the repository root:

    uv run python scripts/docs/generate_reference.py

Writes:

    docs/reference/cli.md         every command, option and default
    docs/reference/models.md      every registered model
    docs/reference/datasets.md    every recorded dataset version
    docs/reference/modules.md     every module, its purpose and public API

Generating beats hand-writing here: a reference that is derived from the code
cannot disagree with it. `--check` re-generates in memory and fails if the
committed files are stale, which is what `scripts/check.sh` runs.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "reference"

# Modules worth documenting, grouped. Keys are dotted module paths.
MODULE_GROUPS: dict[str, list[str]] = {
    "Core": [
        "inmotion",
        "inmotion.cli",
        "inmotion.datasets",
    ],
    "Data": [
        "inmotion.data.paths",
        "inmotion.data.preprocessing",
        "inmotion.data.export",
        "inmotion.data.merge",
        "inmotion.data.clean",
        "inmotion.data.augment",
        "inmotion.data.splits",
    ],
    "Bus network": [
        "inmotion.bus.sources",
        "inmotion.bus.busway",
        "inmotion.bus.aveirobus",
        "inmotion.bus.paths",
        "inmotion.bus.records",
        "inmotion.bus.od.bands",
        "inmotion.bus.od.stops",
        "inmotion.bus.od.tables",
        "inmotion.bus.od.pairs",
        "inmotion.bus.od.matrix",
        "inmotion.bus.od.validate",
        "inmotion.bus.build",
        "inmotion.bus.viz",
        "inmotion.bus.viz_config",
    ],
    "Quantization": [
        "inmotion.quant.spec",
        "inmotion.quant.reference",
        "inmotion.quant.packing",
        "inmotion.quant.model",
        "inmotion.quant.evaluate",
        "inmotion.quant.export",
    ],
    "Models": [
        "inmotion.models.registry",
        "inmotion.models.factories",
        "inmotion.models.checkpoint",
        "inmotion.models.adopt",
        "inmotion.models.wrappers",
    ],
    "CLI": [
        "inmotion.cli",
        "inmotion.cli.model",
        "inmotion.cli.datasets",
        "inmotion.cli.data",
        "inmotion.cli.bus",
        "inmotion.cli.train",
        "inmotion.cli.evaluate",
        "inmotion.cli.analyze",
        "inmotion.cli.plots",
        "inmotion.cli.quantize",
        "inmotion.cli.doctor",
    ],
    "Training pipelines": [
        "inmotion.pipelines.classification",
        "inmotion.pipelines.dl",
        "inmotion.pipelines.dl.config",
        "inmotion.pipelines.dl.devices",
        "inmotion.pipelines.dl.builders",
        "inmotion.pipelines.dl.workers",
        "inmotion.pipelines.dl.pipeline",
        "inmotion.pipelines.exotic",
        "inmotion.pipelines.exotic.config",
        "inmotion.pipelines.exotic.devices",
        "inmotion.pipelines.exotic.data",
        "inmotion.pipelines.exotic.builders",
        "inmotion.pipelines.exotic.training",
        "inmotion.pipelines.exotic.hpo",
        "inmotion.pipelines.exotic.hpo_jepa",
        "inmotion.pipelines.exotic.hpo_sigreg",
        "inmotion.pipelines.exotic.hpo_mamba3",
        "inmotion.pipelines.exotic.pipeline",
        "inmotion.pipelines.ensemble",
        "inmotion.pipelines.mega_ensemble",
        "inmotion.pipelines.hpo_paper",
    ],
    "Evaluation": [
        "inmotion.evaluation.checkpoints",
        "inmotion.evaluation.shap",
        "inmotion.evaluation.feature_importance",
    ],
    "Analysis and figures": [
        "inmotion.analysis.seeds",
        "inmotion.analysis.interference",
        "inmotion.viz.regenerate",
        "inmotion.viz.regenerate_dl",
    ],
    "Deep learning library (inmotion.dl)": [
        "inmotion.dl.config",
        "inmotion.dl.data_loader",
        "inmotion.dl.training",
        "inmotion.dl.optimization",
        "inmotion.dl.evaluation",
        "inmotion.dl.results",
        "inmotion.dl.interference",
        "inmotion.dl.models",
    ],
    "Classical machine learning library (inmotion.classification)": [
        "inmotion.classification.config",
        "inmotion.classification.classifiers",
        "inmotion.classification.training",
        "inmotion.classification.optimization",
        "inmotion.classification.visualization",
        "inmotion.classification.eda",
        "inmotion.classification.data_loader",
    ],
}


def _module_path(dotted: str) -> Path:
    """Filesystem path of an importable module or package."""
    spec = importlib.util.find_spec(dotted)
    if spec is None or spec.origin is None:
        raise ImportError(f"cannot locate {dotted}")
    return Path(spec.origin)


def _summary(doc: str | None) -> str:
    """First sentence-ish line of a docstring."""
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _public_api(tree: ast.Module) -> list[str]:
    """Top-level public classes and functions, with one-line summaries."""
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "func"
            doc = ast.get_docstring(node) or ""
            out.append(f"{kind} `{node.name}`" + (f" - {_summary(doc)}" if doc else ""))
    return out


def gen_modules() -> str:
    lines = [
        "# Module reference",
        "",
        "Generated by `scripts/docs/generate_reference.py` from the source itself,",
        "so it cannot drift from the code. Each entry lists the module's purpose and",
        "its public classes and functions.",
        "",
        "Private names (leading underscore) are omitted; they are implementation",
        "detail.",
        "",
    ]
    for group, modules in MODULE_GROUPS.items():
        lines += [f"## {group}", ""]
        for dotted in modules:
            try:
                path = _module_path(dotted)
            except ImportError:
                lines.append(f"### `{dotted}`\n\nModule not importable in this environment.\n")
                continue
            try:
                source = path.read_text()
                tree = ast.parse(source)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"### `{dotted}`\n\nCould not be parsed: {exc}\n")
                continue

            rel = path.relative_to(REPO_ROOT)
            doc = ast.get_docstring(tree)
            lines.append(f"### `{dotted}`")
            lines.append("")
            lines.append(f"`{rel}` - {len(source.splitlines())} lines")
            lines.append("")
            if doc:
                lines.append(doc.strip())
                lines.append("")
            api = _public_api(tree)
            if api:
                lines.append("Public API:")
                lines.append("")
                lines += [f"- {item}" for item in api]
                lines.append("")
            else:
                lines.append(
                    "No public classes or functions; this module provides data or "
                    "constants consumed elsewhere."
                )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def gen_models() -> str:
    from inmotion.models import registry as reg

    lines = [
        "# Model reference",
        "",
        "Generated from `inmotion.models.registry`. `inmotion model verify` proves",
        "these are loadable; `inmotion model list` prints the same table.",
        "",
        "`features` is what the model consumes at its input: `raw` is the 10 RSSI",
        "values, `rich` is the 18-channel expansion (raw, first and second",
        "differences, window deviation, spectral and shape summaries).",
        "",
        "`params` is the exact module parameter count; `paper` is the figure printed",
        "in the paper's table. Where the two disagree the stored",
        "checkpoint is not the model that table describes, and the gap is recorded in",
        "`PROVENANCE_GAPS`.",
        "",
        f"Canonical model count: **{len(reg.spec_names())}**",
        "",
        "| name | factory | features | params | paper (M) | checkpoint |",
        "|---|---|---|---:|---:|---|",
    ]
    for name in reg.spec_names():
        spec = reg.by_name(name)
        paper = f"{spec.published_m}" if spec.published_m is not None else "-"
        lines.append(
            f"| `{name}` | `{spec.factory}` | {'rich' if spec.rich else 'raw'} | "
            f"{spec.params:,} | {paper} | `{spec.path}` |"
        )

    lines += ["", "## Architecture details", ""]
    for name in reg.spec_names():
        spec = reg.by_name(name)
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- factory: `{spec.factory}`")
        lines.append(f"- checkpoint: `{spec.path}`")
        lines.append(f"- state-dict prefix: `{spec.prefix or '-'}`")
        arch = ", ".join(f"`{k}={v}`" for k, v in sorted(spec.arch.items())) or "(factory defaults)"
        lines.append(f"- architecture: {arch}")
        if spec.note:
            lines.append(f"- note: {spec.note}")
        if spec.name in reg.PROVENANCE_GAPS:
            lines.append(f"- **provenance gap:** {reg.PROVENANCE_GAPS[spec.name]}")
        lines.append("")

    if reg.PROVENANCE_GAPS:
        lines += [
            "## Provenance gaps",
            "",
            "Recorded rather than resolved. `tests/test_checkpoint_contract.py` fails if",
            "one of these is closed by editing the paper figure or the checkpoint count",
            "without restoring the matching artifact.",
            "",
        ]
        for name, note in reg.PROVENANCE_GAPS.items():
            lines += [f"**`{name}`**", "", note, ""]
    return "\n".join(lines).rstrip() + "\n"


def gen_datasets() -> str:
    from inmotion.datasets import build_manifest

    manifest = build_manifest()
    lines = [
        "# Dataset reference",
        "",
        "Generated from `inmotion.datasets`; regenerate the measured fields with",
        "`inmotion datasets manifest --write`. `inmotion datasets verify` re-checks",
        "every file against its recorded checksum.",
        "",
        "The datasets are tracked in git, so their history is the version history.",
        "This table adds what git cannot express: checksums, class balance, lineage,",
        "and whether the labels can be trusted.",
        "",
        f"Canonical training set: **`{manifest['canonical']}`**",
        "",
        "| id | file | rows | devices | classes | status | label audit |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for entry in manifest["datasets"]:
        classes = ", ".join(entry["classes"]) or "-"
        lines.append(
            f"| `{entry['id']}` | `{entry['file']}` | {entry['rows']:,} | "
            f"{entry['devices']} | {classes} | {entry['status']} | {entry['label_audit']} |"
        )

    lines += ["", "## Details", ""]
    for entry in manifest["datasets"]:
        lines.append(f"### `{entry['id']}`")
        lines.append("")
        lines.append(f"- file: `{entry['file']}`")
        lines.append(f"- rows: {entry['rows']:,} ({entry['size_bytes']:,} bytes)")
        lines.append(f"- sha256: `{entry['sha256']}`")
        lines.append(f"- columns: {', '.join(f'`{c}`' for c in entry['columns'])}")
        counts = ", ".join(f"{k}={v:,}" for k, v in sorted(entry["class_counts"].items()))
        lines.append(f"- class counts: {counts}")
        lines.append(f"- devices: {entry['devices']}")
        lines.append(f"- has `synthetic` column: {entry['synthetic']}")
        first = entry["first_commit"]
        lines.append(
            f"- first commit: `{first[:12]}` ({entry['first_commit_date']})"
            if first
            else "- first commit: not recorded"
        )
        lines.append(f"- role: {entry['role']}; status: **{entry['status']}**")
        lines.append(f"- label audit: **{entry['label_audit']}**")
        if entry["validity_note"]:
            lines.append(f"- note: {entry['validity_note']}")
        if entry["lineage"]:
            lin = ", ".join(f"`{k}`={v}" for k, v in entry["lineage"].items())
            lines.append(f"- lineage: {lin}")
        lines.append("")

    lines += [
        "## Schema",
        "",
        "| column | meaning |",
        "|---|---|",
        "| `mac` | device identifier. Retained for cross-device analysis and for",
        "  leave-one-device-out evaluation; **not** a model feature. |",
        "| `1`..`10` | RSSI in dBm, one reading per second for 10 seconds. The only",
        "  raw model features. |",
        "| `label` | route taken: `AA`, `AB`, `BA`, `BB`. The prediction target. |",
        "| `noise` / `noise_label` | whether other devices transmitted concurrently. |",
        "| `concurrent_noise_path` | which route the interfering device was taking, when",
        "  known. Used only by the interference analysis, never as a model input. |",
        "| `synthetic` | padding produced by `inmotion data augment` (v2/v3 only). |",
        "",
        "## Tiers",
        "",
        "| tier | contents | produced by |",
        "|---|---|---|",
        "| `data/raw/` | captures off the access point (Wavecom `.txt`) | field collection |",
        "| `data/interim/` | one CSV per collection session | `inmotion data export` |",
        "| `data/processed/` | merged and augmented training sets |",
        "| | | `inmotion data merge`, `inmotion data augment` |",
        "",
        "`inmotion.data.paths.resolve` searches processed, interim and raw in that",
        "order, so a bare filename still resolves.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def gen_cli() -> str:
    import click
    import typer.main

    from inmotion.cli import app

    root = typer.main.get_command(app)
    lines = [
        "# CLI reference",
        "",
        "Generated from the typer application in `src/inmotion/cli/` by walking the",
        "click command tree, so it lists exactly what the program accepts.",
        "",
        "```",
        "inmotion --help",
        "```",
        "",
    ]

    def render_help(cmd: click.Command, name: str, depth: int) -> None:
        head = "#" * (2 + depth)
        lines.append(f"{head} `{name}`")
        lines.append("")
        help_text = (cmd.help or cmd.short_help or "").strip()
        if help_text:
            lines.append(help_text)
            lines.append("")
        params = getattr(cmd, "params", [])
        opts = [p for p in params if isinstance(p, click.Option) and p.name != "help"]
        args = [p for p in params if isinstance(p, click.Argument)]
        if args:
            lines.append("Arguments:")
            lines.append("")
            for p in args:
                req = "required" if p.required else "optional"
                lines.append(f"- `{p.name.upper()}` ({req}) - {(p.help or '').strip()}")
            lines.append("")
        if opts:
            lines.append("Options:")
            lines.append("")
            for p in opts:
                flag = ", ".join(f"`{o}`" for o in p.opts)
                default = "" if p.default is None else f" (default: `{p.default}`)"
                lines.append(f"- {flag}{default} - {(p.help or '').strip()}")
            lines.append("")
        if isinstance(cmd, click.Group):
            for sub_name in sorted(cmd.commands):
                render_help(cmd.commands[sub_name], f"{name} {sub_name}", depth + 1)

    for name in sorted(root.commands):  # type: ignore[attr-defined]
        render_help(root.commands[name], name, 0)  # type: ignore[attr-defined]

    lines += [
        "## Pipeline modules",
        "",
        "Every training, evaluation and analysis pipeline exposes the same shape:",
        "a typed `RunConfig` dataclass and a `run(args) -> int` function. The CLI",
        "builds the config from typed options, so the pipelines contain no argument",
        "parsing and can be driven from Python directly:",
        "",
        "```python",
        "from inmotion.pipelines.exotic import RunConfig, run",
        "",
        "run(RunConfig(model='ts_jepa', seed=42, pretrain_epochs=500))",
        "```",
        "",
        "The two largest pipelines are packages with one module per responsibility:",
        "",
        "| pipeline | package | modules |",
        "|---|---|---|",
        "| `inmotion train dl` | `inmotion.pipelines.dl` |",
        "| | | config, devices, builders, workers, pipeline |",
        "| `inmotion train exotic` | `inmotion.pipelines.exotic` |",
        "| | | config, devices, data, builders, training, |",
        "| | | hpo, hpo_jepa, hpo_sigreg, hpo_mamba3, pipeline |",
        "",
        "`python -m inmotion.pipelines.<name>` runs the same CLI for those packages.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


# Each entry: (command path, pipeline module, RunConfig class name)
PIPELINE_CONFIGS: list[tuple[str, str, str]] = [
    ("train dl", "inmotion.pipelines.dl", "RunConfig"),
    ("train exotic", "inmotion.pipelines.exotic", "RunConfig"),
    ("train classification", "inmotion.pipelines.classification", "RunConfig"),
    ("train hpo-paper", "inmotion.pipelines.hpo_paper", "RunConfig"),
    ("evaluate ensemble", "inmotion.pipelines.ensemble", "RunConfig"),
    ("evaluate mega-ensemble", "inmotion.pipelines.mega_ensemble", "RunConfig"),
    ("evaluate checkpoints", "inmotion.evaluation.checkpoints", "RunConfig"),
    ("evaluate shap", "inmotion.evaluation.shap", "RunConfig"),
    ("evaluate feature-importance", "inmotion.evaluation.feature_importance", "RunConfig"),
    ("analyze seeds", "inmotion.analysis.seeds", "RunConfig"),
    ("analyze interference", "inmotion.analysis.interference", "RunConfig"),
    ("plots classical", "inmotion.viz.regenerate", "RunConfig"),
    ("plots dl", "inmotion.viz.regenerate_dl", "RunConfig"),
    ("data clean", "inmotion.data.clean", "RunConfig"),
]


def gen_pipelines() -> str:
    """Every pipeline's option surface, read from its RunConfig dataclass."""
    import dataclasses
    import importlib

    lines = [
        "# Pipeline reference",
        "",
        "Generated from the pipelines' own `RunConfig` dataclasses, so the option",
        "list cannot drift from what the code accepts.",
        "",
        "Every pipeline has the same shape: a typed config and a `run(args) -> int`.",
        "The CLI builds the config from the options below; from Python you construct",
        "it directly:",
        "",
        "```python",
        "from inmotion.pipelines.exotic import RunConfig, run",
        "",
        "run(RunConfig(model='ts_jepa', seed=42, pretrain_epochs=500))",
        "```",
        "",
        "Return value is the exit code, so a caller can check it.",
        "",
        "| command | pipeline module | options |",
        "|---|---|---:|",
    ]
    loaded: list[tuple[str, str, type]] = []
    for command, module_name, cls_name in PIPELINE_CONFIGS:
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        loaded.append((command, module_name, cls))
        lines.append(f"| `inmotion {command}` | `{module_name}` | {len(dataclasses.fields(cls))} |")

    for command, module_name, cls in loaded:
        lines += ["", f"## `inmotion {command}`", "", f"Module: `{module_name}`", ""]
        doc = (cls.__doc__ or "").strip().splitlines()
        if doc:
            lines += [doc[0], ""]
        lines += [
            "| option | type | default |",
            "|---|---|---|",
        ]
        for field in dataclasses.fields(cls):
            annotation = str(field.type).replace("NoneType", "None")
            annotation = annotation.replace("inmotion.", "")
            if field.default is not dataclasses.MISSING:
                default = f"`{field.default!r}`"
            elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                try:
                    default = f"`{field.default_factory()!r}`"  # type: ignore[misc]
                except Exception:  # noqa: BLE001
                    default = "(computed)"
            else:
                default = ""
            lines.append(f"| `--{field.name.replace('_', '-')}` | `{annotation}` | {default} |")
        # note which fields map to a CLI flag whose name differs
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


GENERATORS = {
    "cli.md": gen_cli,
    "pipelines.md": gen_pipelines,
    "models.md": gen_models,
    "datasets.md": gen_datasets,
    "modules.md": gen_modules,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the committed files are stale."
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for filename, generate in GENERATORS.items():
        content = generate()
        path = OUT_DIR / filename
        if args.check:
            existing = path.read_text() if path.exists() else ""
            if existing != content:
                stale.append(str(path.relative_to(REPO_ROOT)))
        else:
            path.write_text(content)
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    if args.check and stale:
        print("stale reference docs (run scripts/docs/generate_reference.py):")
        for s in stale:
            print("  ", s)
        return 1
    if args.check:
        print("reference docs are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

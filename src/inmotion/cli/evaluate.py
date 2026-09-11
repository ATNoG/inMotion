"""`inmotion evaluate` — evaluate and combine models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.cli._common import existing_file, run_pipeline
from inmotion.data.paths import processed

app = typer.Typer(no_args_is_help=True, help="Evaluate and combine models.")
NAME = "evaluate"
HELP = "Evaluate and combine models."


@app.command("ensemble", help="Greedy ensemble selection over existing checkpoints.")
def ensemble(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    rich: Annotated[bool, typer.Option("--rich", help="Use the 18-channel feature set.")] = False,
) -> None:
    """Combine saved checkpoints with Caruana forward selection."""
    from inmotion.pipelines.ensemble import RunConfig, run

    run_pipeline(run, RunConfig(data=data, seed=seed, rich=rich))


@app.command("mega-ensemble", help="OOF stacking over every model family, with calibration.")
def mega_ensemble(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    size_frontier: Annotated[
        bool, typer.Option("--size-frontier", help="Report the ensemble-size frontier.")
    ] = False,
    no_tabpfn: Annotated[bool, typer.Option("--no-tabpfn", help="Exclude TabPFN.")] = False,
    no_classical: Annotated[
        bool, typer.Option("--no-classical", help="Exclude the classical ML members.")
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", help="Write the result table here.")
    ] = None,
) -> None:
    """Combine DL, JEPA, classical ML and TabPFN members into one model."""
    from inmotion.pipelines.mega_ensemble import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            data=data,
            seed=seed,
            size_frontier=size_frontier,
            no_tabpfn=no_tabpfn,
            no_classical=no_classical,
            output=output,
        ),
    )


@app.command("checkpoints", help="Retroactive per-class metrics for saved DL checkpoints.")
def checkpoints(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    checkpoints_dir: Annotated[
        Path, typer.Option("--checkpoints-dir", help="Directory of .pt checkpoints.")
    ] = Path("models/dl"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Where to write the results.")
    ] = Path("results/dl"),
    optuna_db: Annotated[
        str | None, typer.Option("--optuna-db", help="Optuna storage URL, for HPO checkpoints.")
    ] = None,
    no_wandb: Annotated[bool, typer.Option("--no-wandb", help="Disable WandB logging.")] = False,
) -> None:
    """Evaluate every checkpoint in a directory with extended metrics."""
    from inmotion.evaluation.checkpoints import RunConfig, run

    cfg = RunConfig(
        data=data,
        seed=seed,
        checkpoints_dir=checkpoints_dir,
        output_dir=output_dir,
        no_wandb=no_wandb,
    )
    if optuna_db is not None:
        cfg.optuna_db = optuna_db
    run_pipeline(run, cfg)


@app.command("shap", help="SHAP channel attribution for the world models.")
def shap(
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="Model checkpoint to explain.")
    ] = Path("backup/ts_jepa_ft_seed42.pt"),
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    model: Annotated[
        str,
        typer.Option("--model", help="Model family: t_jepa, ts_jepa, lejepa, cf_jepa or sigreg."),
    ] = "ts_jepa",
    samples: Annotated[int, typer.Option("--samples", help="Number of samples to explain.")] = 50,
    nsamples: Annotated[int, typer.Option("--nsamples", help="SHAP evaluation budget.")] = 150,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Figure output directory.")
    ] = Path("plots/shap"),
) -> None:
    """Attribute predictions to the 18 feature channels."""
    from inmotion.evaluation.shap import RunConfig, run

    existing_file(checkpoint, "checkpoint")
    run_pipeline(
        run,
        RunConfig(
            checkpoint=checkpoint,
            data=data,
            model=model,
            samples=samples,
            nsamples=nsamples,
            seed=seed,
            output_dir=output_dir,
        ),
    )


@app.command("feature-importance", help="Permutation feature importance for the DL models.")
def feature_importance(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    epochs: Annotated[int, typer.Option("--epochs", help="Epochs per model.")] = 100,
    patience: Annotated[int, typer.Option("--patience", help="Early-stopping patience.")] = 20,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Figure output directory.")
    ] = Path("docs/icaisf/paper/images"),
    checkpoints_dir: Annotated[
        Path, typer.Option("--checkpoints-dir", help="Directory of DL checkpoints.")
    ] = Path("models/dl"),
) -> None:
    """Shuffle each feature channel and measure the MCC drop."""
    from inmotion.evaluation.feature_importance import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            data=data,
            seed=seed,
            epochs=epochs,
            patience=patience,
            output_dir=output_dir,
            checkpoints_dir=checkpoints_dir,
        ),
    )

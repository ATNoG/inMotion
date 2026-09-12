"""`inmotion train` — train models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.cli._common import run_pipeline
from inmotion.data.paths import processed

app = typer.Typer(no_args_is_help=True, help="Train models.")
NAME = "train"
HELP = "Train models."


@app.command("dl", help="Supervised deep learning: baselines, HPO/NAS, MoE, DeepStack.")
def dl(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    epochs: Annotated[int, typer.Option("--epochs", help="Max epochs per model.")] = 150,
    trials: Annotated[int, typer.Option("--trials", help="Optuna trials per study.")] = 50,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size.")] = 64,
    num_gpus: Annotated[int, typer.Option("--num-gpus", help="GPUs to use; 0 = auto-detect.")] = 0,
    models_dir: Annotated[
        Path | None, typer.Option("--models-dir", help="Checkpoint output directory.")
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Results CSV directory.")
    ] = None,
    plots_dir: Annotated[
        Path | None, typer.Option("--plots-dir", help="Figure output directory.")
    ] = None,
    wandb_project: Annotated[
        str | None, typer.Option("--wandb-project", help="Override the WandB project.")
    ] = None,
    no_wandb: Annotated[bool, typer.Option("--no-wandb", help="Disable WandB logging.")] = False,
    no_optuna: Annotated[
        bool, typer.Option("--no-optuna", help="Skip HPO and NAS studies.")
    ] = False,
    no_moe: Annotated[
        bool, typer.Option("--no-moe", help="Skip the mixture-of-experts phase.")
    ] = False,
    no_deepstack: Annotated[
        bool, typer.Option("--no-deepstack", help="Skip the DeepStack phase.")
    ] = False,
    no_meta: Annotated[
        bool, typer.Option("--no-meta", help="Skip the metadata-fusion phases.")
    ] = False,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="Retrain even when a checkpoint exists.")
    ] = False,
) -> None:
    """Train the supervised deep-learning models and their ensembles."""
    from inmotion.pipelines.dl import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            data=data,
            seed=seed,
            epochs=epochs,
            trials=trials,
            batch_size=batch_size,
            num_gpus=num_gpus,
            models_dir=models_dir,
            results_dir=results_dir,
            plots_dir=plots_dir,
            wandb_project=wandb_project,
            no_wandb=no_wandb,
            no_optuna=no_optuna,
            no_moe=no_moe,
            no_deepstack=no_deepstack,
            no_meta=no_meta,
            no_resume=no_resume,
        ),
    )


@app.command("exotic", help="JEPA world models and SIGReg: SSL pretraining and fine-tuning.")
def exotic(
    model: Annotated[
        str,
        typer.Option(
            "--model", help="t_jepa, ts_jepa, lejepa, cf_jepa, sigreg or mamba3_<variant>."
        ),
    ] = "t_jepa",
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size.")] = 128,
    gpu: Annotated[int, typer.Option("--gpu", help="GPU device index.")] = 0,
    rich_features: Annotated[
        bool, typer.Option("--rich-features", help="Use the 18-channel feature set.")
    ] = False,
    pretrain_epochs: Annotated[
        int, typer.Option("--pretrain-epochs", help="Stage-1 epochs.")
    ] = 300,
    finetune_epochs: Annotated[int, typer.Option("--finetune-epochs", help="Stage-2 epochs.")] = 50,
    epochs: Annotated[
        int, typer.Option("--epochs", help="Epochs for directly-supervised models.")
    ] = 150,
    patience: Annotated[int, typer.Option("--patience", help="Early-stopping patience.")] = 25,
    lr: Annotated[
        float, typer.Option("--lr", help="Learning rate for directly-supervised models.")
    ] = 1e-3,
    pretrain_lr: Annotated[
        float, typer.Option("--pretrain-lr", help="Stage-1 learning rate.")
    ] = 3e-4,
    finetune_lr: Annotated[
        float, typer.Option("--finetune-lr", help="Stage-2 learning rate.")
    ] = 1e-3,
    probe_cadence: Annotated[
        int, typer.Option("--probe-cadence", help="Epochs between linear probes.")
    ] = 10,
    probe_patience: Annotated[
        int, typer.Option("--probe-patience", help="Probe-based patience.")
    ] = 15,
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="Resume from a pretrained encoder.")
    ] = None,
    finetune_only: Annotated[
        bool, typer.Option("--finetune-only", help="Skip pretraining.")
    ] = False,
    d_model: Annotated[int, typer.Option("--d-model", help="Model width.")] = 256,
    nhead: Annotated[int, typer.Option("--nhead", help="Attention heads.")] = 8,
    num_layers: Annotated[int, typer.Option("--num-layers", help="Transformer layers.")] = 4,
    dim_ff: Annotated[int, typer.Option("--dim-ff", help="Feed-forward width.")] = 512,
    pred_dim: Annotated[int, typer.Option("--pred-dim", help="Predictor width.")] = 128,
    pred_layers: Annotated[int, typer.Option("--pred-layers", help="Predictor depth.")] = 2,
    d_state: Annotated[int, typer.Option("--d-state", help="Mamba-3 state width.")] = 16,
    dropout: Annotated[
        float, typer.Option("--dropout", help="Dropout for the Mamba-3 hybrids.")
    ] = 0.2,
    mimo_rank: Annotated[int, typer.Option("--mimo-rank", help="Mamba-3 MIMO rank.")] = 4,
    num_filters: Annotated[
        int, typer.Option("--num-filters", help="SIGReg convolutional filters.")
    ] = 192,
    num_blocks: Annotated[int, typer.Option("--num-blocks", help="SIGReg residual blocks.")] = 3,
    latent_dim: Annotated[int, typer.Option("--latent-dim", help="SIGReg latent width.")] = 128,
    sigreg_lambda: Annotated[
        float, typer.Option("--sigreg-lambda", help="SIGReg regulariser weight.")
    ] = 0.01,
    hpo: Annotated[bool, typer.Option("--hpo", help="Run an Optuna study first.")] = False,
    hpo_trials: Annotated[int, typer.Option("--hpo-trials", help="Trials per study.")] = 30,
    hpo_objective: Annotated[
        str, typer.Option("--hpo-objective", help="mcc, efficiency or pareto.")
    ] = "mcc",
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Checkpoint output directory.")
    ] = Path("models/exotic"),
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Results CSV directory.")
    ] = Path("results/exotic"),
    viz_dir: Annotated[Path, typer.Option("--viz-dir", help="Figure output directory.")] = Path(
        "models/exotic/viz"
    ),
    no_wandb: Annotated[bool, typer.Option("--no-wandb", help="Disable WandB logging.")] = False,
    wandb_project: Annotated[
        str, typer.Option("--wandb-project", help="WandB project name.")
    ] = "inMotion-exotic-hpo",
) -> None:
    """Pretrain and fine-tune a world model, or train SIGReg / a Mamba-3 hybrid."""
    from inmotion.pipelines.exotic import RunConfig, run

    cfg = RunConfig(
        model=model,
        data=data,
        seed=seed,
        batch_size=batch_size,
        gpu=gpu,
        rich_features=rich_features,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        epochs=epochs,
        patience=patience,
        lr=lr,
        pretrain_lr=pretrain_lr,
        finetune_lr=finetune_lr,
        probe_cadence=probe_cadence,
        probe_patience=probe_patience,
        checkpoint=checkpoint,
        finetune_only=finetune_only,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        pred_dim=pred_dim,
        pred_layers=pred_layers,
        d_state=d_state,
        dropout=dropout,
        mimo_rank=mimo_rank,
        num_filters=num_filters,
        num_blocks=num_blocks,
        latent_dim=latent_dim,
        sigreg_lambda=sigreg_lambda,
        hpo=hpo,
        hpo_trials=hpo_trials,
        hpo_objective=hpo_objective,
        models_dir=models_dir,
        results_dir=results_dir,
        viz_dir=viz_dir,
        no_wandb=no_wandb,
        wandb_project=wandb_project,
    )
    run_pipeline(run, cfg)


@app.command("classification", help="Classical ML classifiers: EDA, CV, Optuna, importance.")
def classification(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    optimize: Annotated[
        bool, typer.Option("--optimize", help="Run the Optuna tuning stage.")
    ] = False,
    dashboard: Annotated[
        bool, typer.Option("--dashboard", help="Launch optuna-dashboard after tuning.")
    ] = False,
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Model output directory.")
    ] = Path("models"),
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Results output directory.")
    ] = Path("results"),
    plots_dir: Annotated[Path, typer.Option("--plots-dir", help="Figure output directory.")] = Path(
        "plots"
    ),
) -> None:
    """Run the classical machine-learning comparison."""
    from inmotion.pipelines.classification import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            data=data,
            seed=seed,
            optimize=optimize,
            dashboard=dashboard,
            models_dir=models_dir,
            results_dir=results_dir,
            plots_dir=plots_dir,
        ),
    )


@app.command("hpo-paper", help="Retrain the paper's HPO-best configurations.")
def hpo_paper(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV.")] = processed("dataset.csv"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    epochs: Annotated[int, typer.Option("--epochs", help="Epochs per model.")] = 150,
    patience: Annotated[int, typer.Option("--patience", help="Early-stopping patience.")] = 20,
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated subset, e.g. gru,tcn.")
    ] = None,
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Checkpoint output directory.")
    ] = Path("models/dl"),
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Results output directory.")
    ] = Path("results/dl"),
) -> None:
    """Rebuild the HPO-best models, since the Optuna databases were deleted."""
    from inmotion.cli._common import parse_csv_list
    from inmotion.pipelines.hpo_paper import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            data=data,
            seed=seed,
            epochs=epochs,
            patience=patience,
            models=parse_csv_list(models) or None,
            models_dir=models_dir,
            results_dir=results_dir,
        ),
    )

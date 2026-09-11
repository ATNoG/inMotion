"""DL pipeline configuration: model lists and run options."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inmotion.data.paths import processed

SINGLE_MODEL_NAMES: list[str] = [
    "RNN",
    "GRU",
    "LSTM",
    "CNN",
    "TCN",
    # "Transformer",     # consistently worst; uncomment if needed
    # "BiLSTM",          # bottom-2 in augmented + noise
    # "CNN2DGRU",        # strictly worse than CNN2DLSTM everywhere
    "CNN2DLSTM",
    "Mamba",
]

HPO_TYPES: list[str] = [
    # "rnn",             # weakest HPO by avg cross-dataset MCC
    "gru",
    "lstm",
    "cnn",
    "tcn",
    "transformer",
    "bilstm",
    # "cnn2drnn",        # 2nd weakest HPO; redundant to cnn + rnn separately
    "mamba",
]

META_MODEL_NAMES: list[str] = ["MetaFusion_LSTM", "MetaFusion_GRU"]

DS_BASE_NAMES: list[str] = [
    "DS_RNN_A",
    "DS_RNN_B",
    "DS_GRU_A",
    # "DS_GRU_B",
    # "DS_LSTM_A",
    # "DS_LSTM_B",
    "DS_CNN_A",
    "DS_CNN_B",
    "DS_TCN_A",
    "DS_TCN_B",
    # "DS_Transformer_A",
    # "DS_Transformer_B",
    # "DS_BiLSTM_A",
    # "DS_BiLSTM_B",
    # "DS_CNN2DRNN_A",
    "DS_CNN2DRNN_B",
    "DS_Mamba_A",
    "DS_Mamba_B",
]


@dataclass
class RunConfig:
    """Typed options for the DL pipeline.

    Field names match the argparse namespace this replaced, so the pipeline body
    is unchanged. Built by the CLI (`inmotion train dl`).
    """

    seed: int = 42
    epochs: int = 150
    trials: int = 50
    batch_size: int = 64
    num_gpus: int = 0
    no_wandb: bool = False
    no_optuna: bool = False
    no_moe: bool = False
    no_deepstack: bool = False
    no_meta: bool = False
    no_resume: bool = False
    wandb_project: str | None = None
    data: Path = field(default_factory=lambda: processed("dataset.csv"))
    models_dir: Path | None = None
    results_dir: Path | None = None
    plots_dir: Path | None = None

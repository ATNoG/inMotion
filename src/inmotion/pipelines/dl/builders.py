"""Model construction for the DL pipeline."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from inmotion.dl.config import DLConfig
from inmotion.dl.models.bilstm import BiLSTMClassifier
from inmotion.dl.models.cnn import CNNClassifier
from inmotion.dl.models.cnn2d_rnn import CNN2DRNNClassifier
from inmotion.dl.models.gru import GRUClassifier
from inmotion.dl.models.lstm import LSTMClassifier
from inmotion.dl.models.mamba import MambaClassifier
from inmotion.dl.models.rnn import RNNClassifier
from inmotion.dl.models.tcn import TCNClassifier
from inmotion.dl.models.transformer import TransformerClassifier


def _build_model_by_name(name: str, config: DLConfig) -> nn.Module:
    base_kw: dict[str, object] = {
        "in_features": config.in_features,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_classes": config.num_classes,
        "dropout": config.dropout,
    }
    match name:
        case "RNN":
            return RNNClassifier(**base_kw)  # type: ignore[arg-type]
        case "GRU":
            return GRUClassifier(**base_kw)  # type: ignore[arg-type]
        case "LSTM":
            return LSTMClassifier(**base_kw, use_attention=True)  # type: ignore[arg-type]
        case "BiLSTM":
            return BiLSTMClassifier(**base_kw, use_attention=True)  # type: ignore[arg-type]
        case "CNN":
            return CNNClassifier(
                in_features=config.in_features,
                num_filters=128,
                num_blocks=3,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "TCN":
            return TCNClassifier(
                in_features=config.in_features,
                num_channels=256,
                kernel_size=3,
                depth=6,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "Transformer":
            return TransformerClassifier(
                in_features=config.in_features,
                d_model=128,
                n_heads=8,
                num_layers=4,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "CNN2DLSTM":
            return CNN2DRNNClassifier(
                in_features=config.in_features,
                num_filters=64,
                cnn_depth=3,
                hidden_size=config.hidden_size,
                num_rnn_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                rnn_type="lstm",
            )
        case "CNN2DGRU":
            return CNN2DRNNClassifier(
                in_features=config.in_features,
                num_filters=64,
                cnn_depth=3,
                hidden_size=config.hidden_size,
                num_rnn_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                rnn_type="gru",
            )
        case "Mamba":
            return MambaClassifier(
                in_features=config.in_features,
                d_model=config.d_model,
                d_state=16,
                num_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                mimo_rank=4,
            )
        case "T-JEPA":
            # T-JEPA requires pretraining before classification.
            # This builds the classifier with a randomly-initialized encoder;
            # pretraining should be done separately via pretrain_t_jepa.py.
            from inmotion.dl.models.t_jepa import TJEPAClassifier, TJEPAModel

            pretrained = TJEPAModel(
                n_features=10,
                d_model=config.d_model,
                nhead=4,
                num_layers=config.num_layers,
                dim_feedforward=256,
                n_reg_tokens=2,
                pred_dim=64,
                pred_num_layers=2,
            )
            return TJEPAClassifier(
                pretrained=pretrained,
                num_classes=config.num_classes,
                pooling="mean",
                dropout=config.dropout,
            )
        case _:
            raise ValueError(f"Unknown model: {name}")


def _build_ds_variant(name: str, config: DLConfig) -> nn.Module:
    """Build one of the Deep-Stack base model variants."""
    match name:
        case "DS_RNN_A":
            return RNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_RNN_B":
            return RNNClassifier(
                config.in_features, 256, 4, config.num_classes, 0.3, bidirectional=True
            )
        case "DS_GRU_A":
            return GRUClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_GRU_B":
            return GRUClassifier(
                config.in_features, 256, 3, config.num_classes, 0.3, bidirectional=True
            )
        case "DS_LSTM_A":
            return LSTMClassifier(config.in_features, 64, 2, config.num_classes, 0.2, False, True)
        case "DS_LSTM_B":
            return LSTMClassifier(config.in_features, 256, 3, config.num_classes, 0.3, True, True)
        case "DS_CNN_A":
            return CNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2, [3, 5])
        case "DS_CNN_B":
            return CNNClassifier(config.in_features, 256, 4, config.num_classes, 0.3, [3, 5, 7, 9])
        case "DS_TCN_A":
            return TCNClassifier(config.in_features, 128, 3, 4, config.num_classes, 0.2)
        case "DS_TCN_B":
            return TCNClassifier(config.in_features, 512, 7, 8, config.num_classes, 0.3)
        case "DS_Transformer_A":
            return TransformerClassifier(config.in_features, 64, 4, 2, config.num_classes, 0.1)
        case "DS_Transformer_B":
            return TransformerClassifier(config.in_features, 256, 8, 4, config.num_classes, 0.15)
        case "DS_BiLSTM_A":
            return BiLSTMClassifier(config.in_features, 64, 2, config.num_classes, 0.2, True)
        case "DS_BiLSTM_B":
            return BiLSTMClassifier(config.in_features, 256, 3, config.num_classes, 0.3, True)
        case "DS_CNN2DRNN_A":
            return CNN2DRNNClassifier(
                config.in_features, 32, 2, 64, 2, config.num_classes, 0.2, "lstm"
            )
        case "DS_CNN2DRNN_B":
            return CNN2DRNNClassifier(
                config.in_features,
                128,
                3,
                256,
                3,
                config.num_classes,
                0.3,
                "lstm",
                bidirectional=True,
            )
        case "DS_Mamba_A":
            return MambaClassifier(
                config.in_features, 64, 16, 2, 2, config.num_classes, 0.2, mimo_rank=4
            )
        case "DS_Mamba_B":
            return MambaClassifier(
                config.in_features, 256, 32, 4, 2, config.num_classes, 0.3, mimo_rank=8
            )
        case _:
            raise ValueError(f"Unknown DS variant: {name}")


def _extract_logits(
    model: nn.Module,
    loader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    device: torch.device,
) -> np.ndarray:
    """Run model on loader, return concatenated logits (N, num_classes)."""
    model.eval().to(device)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            parts.append(model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)


def _build_hpo_best_model(model_type: str, config: DLConfig, params: dict) -> nn.Module:
    """Rebuild the best Optuna model from stored HPO trial params."""
    dropout = float(params.get("dropout", config.dropout))
    kernel_sets: list[list[int]] = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 9]]
    if model_type == "rnn":
        return RNNClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("bidirectional", False)),
        )
    elif model_type == "gru":
        return GRUClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("bidirectional", False)),
        )
    elif model_type == "lstm":
        return LSTMClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("bidirectional", False)),
            bool(params.get("use_attention", False)),
        )
    elif model_type == "bilstm":
        return BiLSTMClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("use_attention", True)),
        )
    elif model_type == "cnn":
        ks = kernel_sets[int(params.get("kernel_set", 0))]
        return CNNClassifier(
            config.in_features,
            int(params.get("num_filters", 64)),
            int(params.get("num_blocks", 2)),
            config.num_classes,
            dropout,
            ks,
        )
    elif model_type == "tcn":
        return TCNClassifier(
            config.in_features,
            int(params.get("num_channels", 128)),
            int(params.get("kernel_size", 3)),
            int(params.get("depth", 4)),
            config.num_classes,
            dropout,
        )
    elif model_type == "transformer":
        d_model = int(params.get("d_model", 64))
        n_heads = int(params.get("n_heads", 4))
        while d_model % n_heads != 0:
            n_heads = max(1, n_heads // 2)
        return TransformerClassifier(
            config.in_features,
            d_model,
            n_heads,
            int(params.get("num_layers", 2)),
            config.num_classes,
            dropout,
        )
    elif model_type == "cnn2drnn":
        return CNN2DRNNClassifier(
            in_features=config.in_features,
            num_filters=int(params.get("num_filters", 32)),
            cnn_depth=int(params.get("cnn_depth", 2)),
            hidden_size=int(params.get("hidden_size", config.hidden_size)),
            num_rnn_layers=int(params.get("num_rnn_layers", 2)),
            num_classes=config.num_classes,
            dropout=dropout,
            rnn_type=str(params.get("rnn_type", "lstm")),
            bidirectional=bool(params.get("bidirectional", False)),
        )
    elif model_type == "mamba":
        return MambaClassifier(
            in_features=config.in_features,
            d_model=int(params.get("d_model", config.d_model)),
            d_state=int(params.get("d_state", 16)),
            num_layers=int(params.get("num_layers", config.num_layers)),
            expand=int(params.get("expand", 2)),
            num_classes=config.num_classes,
            dropout=dropout,
            mimo_rank=int(params.get("mimo_rank", 4)),
        )
    else:
        raise ValueError(f"Unknown HPO model_type: {model_type!r}")

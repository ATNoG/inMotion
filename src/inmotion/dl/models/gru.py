"""GRU classifier with optional self-attention pooling over hidden states."""

import torch
from torch import Tensor, nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        if use_attention:
            self.attn = nn.Linear(out_size, 1)
        self.norm = nn.LayerNorm(out_size)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)  # (B, T, out_size)
        if self.use_attention:
            scores = self.attn(out).squeeze(-1)  # (B, T)
            weights = torch.softmax(scores, dim=-1)  # (B, T)
            context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, out_size)
        else:
            context = out[:, -1, :]
        context = self.norm(context)
        context = self.drop(context)
        return self.classifier(context)

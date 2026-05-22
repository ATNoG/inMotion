"""Bidirectional LSTM classifier with self-attention pooling.

Dedicated BiLSTM class (bidirectional=True is forced) so it appears as a
distinct architecture in HPO / ensemble comparisons.
"""

import torch
from torch import Tensor, nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        out_size = hidden_size * 2  # always bidirectional
        if use_attention:
            self.attn = nn.Linear(out_size, 1)
        self.norm = nn.LayerNorm(out_size)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)  # (B, T, hidden*2)
        if self.use_attention:
            scores = self.attn(out).squeeze(-1)  # (B, T)
            weights = torch.softmax(scores, dim=-1)
            context = (out * weights.unsqueeze(-1)).sum(dim=1)
        else:
            # Concat last forward and first backward hidden state
            context = torch.cat(
                [out[:, -1, : out.size(-1) // 2], out[:, 0, out.size(-1) // 2 :]], dim=-1
            )
        context = self.norm(context)
        context = self.drop(context)
        return self.classifier(context)

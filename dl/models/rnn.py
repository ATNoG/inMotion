"""Vanilla RNN classifier."""

from torch import Tensor, nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        self.drop = nn.Dropout(dropout)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, in_features)
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # (B, hidden)
        last = self.norm(last)
        last = self.drop(last)
        return self.classifier(last)  # (B, num_classes)

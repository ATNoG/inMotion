"""Multi-scale 1D CNN classifier with residual connections."""

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class CNNClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_filters: int,
        num_blocks: int,
        num_classes: int,
        dropout: float = 0.3,
        kernel_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        # Multi-scale input convolutions
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_features, num_filters // len(kernel_sizes), k, padding=k // 2),
                    nn.BatchNorm1d(num_filters // len(kernel_sizes)),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )
        branch_out = (num_filters // len(kernel_sizes)) * len(kernel_sizes)

        # Projection to num_filters
        self.proj = nn.Sequential(
            nn.Conv1d(branch_out, num_filters, 1),
            nn.BatchNorm1d(num_filters),
            nn.GELU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, 3, dropout) for _ in range(num_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, in_features) → (B, in_features, seq_len)
        x = x.permute(0, 2, 1)
        branch_outs = [b(x) for b in self.branches]
        x = torch.cat(branch_outs, dim=1)  # (B, branch_out, seq_len)
        x = self.proj(x)
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)  # (B, num_filters)
        x = self.drop(x)
        return self.classifier(x)

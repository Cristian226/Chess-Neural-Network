import torch
from torch import nn


IN_CHANNELS = 20
CHANNELS = 128
NUM_BLOCKS = 8
DROPOUT = 0.2
SE_REDUCTION = 4


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = SE_REDUCTION):
        super().__init__()
        reduced = channels // reduction

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),

            SEBlock(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ChessEvalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True),
            *[ResidualBlock(CHANNELS) for _ in range(NUM_BLOCKS)],
        )

        self.head = nn.Sequential(
            nn.Conv2d(CHANNELS, CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(CHANNELS * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        return self.head(x)

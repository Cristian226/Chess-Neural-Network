import torch
from torch import nn


IN_CHANNELS = 18
CHANNELS = 128
NUM_BLOCKS = 4
DROPOUT = 0.2


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


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

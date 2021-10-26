from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from src.config.hyperparameters import UNET_RESNET_FEATURES


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: int = 1,
        padding: Union[str, int] = "same",
    ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_factor: int = 4, num_repeats: int = 1) -> None:
        super().__init__()
        self.interim_channels = in_channels // reduction_factor
        layers = []
        for _ in range(num_repeats):
            layers.extend(
                [
                    ConvBlock(in_channels=in_channels, out_channels=self.interim_channels, kernel_size=1),
                    ConvBlock(in_channels=self.interim_channels, out_channels=self.interim_channels),
                    ConvBlock(in_channels=self.interim_channels, out_channels=in_channels, kernel_size=1),
                ]
            )

        self.resnet_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.resnet_block(x)


class UNETConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_repeats: int = 1) -> None:
        super().__init__()
        self.unet_conv_block = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            ResNetBlock(in_channels=out_channels, num_repeats=num_repeats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet_conv_block(x)


class UNET(nn.Module):
    def __init__(self, features: list[tuple[int, int]], in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.contractions = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.expansions = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contraction part
        for feature, num_repeats in features:
            self.contractions.append(UNETConvBlock(in_channels=in_channels, out_channels=feature, num_repeats=num_repeats))
            in_channels = feature

        # "Bottleneck" part
        self.bottleneck = UNETConvBlock(in_channels=feature, out_channels=feature * 2)

        # Expansion part
        for feature, num_repeats in features[::-1]:
            self.upsamplings.append(
                nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2)
            )
            self.expansions.append(
                UNETConvBlock(in_channels=feature * 2, out_channels=feature, num_repeats=num_repeats),
            )

        # Final part
        self.final_conv = nn.Conv2d(in_channels=feature, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for contraction in self.contractions:
            x = contraction(x)
            skip_connections.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)

        for upsampling, expansion in zip(self.upsamplings, self.expansions):
            x = upsampling(x)
            skip_connection = skip_connections.pop()

            # tensor with odd size (e.g. 97*97) after maxpool will have
            # even size (48*48), so concatination might fail. For that reason
            # resize is performed
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = expansion(x)

        return self.final_conv(x)


def test_architecture():
    x = torch.randn((1, 1, 388, 388))
    model = UNET(features=UNET_RESNET_FEATURES, in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test_architecture()
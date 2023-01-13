from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.config.hyperparameters import UNET_FEATURES


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: int = 1,
        padding: Union[str, int] = "same",
    ) -> None:
        """Convolution block that contains 2d Convolution with Batch Normalization and ReLU activation function.

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : Union[int, tuple[int, int]], optional
            size of the kernel/filter for convolution, by default 3
        stride : int, optional
            size of the stride (step) for convolution, by default 1
        padding : Union[str, int], optional
            padding for convolution, if no padding the output size will be smaller than input, by default "same"
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.conv_block(x)


class UNETConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """U-Net specific convolution block that has two convolution blocks inside.

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.double_conv(x)


class UNET(nn.Module):
    def __init__(self, features: list[int], in_channels: int = 3, out_channels: int = 1) -> None:
        """Full U-Net model.

        Parameters
        ----------
        features : list[int]
            list of number of features in each U-Net block
        in_channels : int, optional
            number of input channels - how many channels images have, by default 3
        out_channels : int, optional
            number of output channels - how many channels masks have, by default 1
        """
        super().__init__()
        self.contraction_list = nn.ModuleList()
        self.upsampling_list = nn.ModuleList()
        self.expansions = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contraction part
        for feature in features:
            self.contraction_list.append(UNETConvBlock(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # "Bottleneck" part
        self.bottleneck = UNETConvBlock(in_channels=feature, out_channels=feature * 2)

        # Expansion part
        for feature in features[::-1]:
            self.upsampling_list.append(
                nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2),
            )
            self.expansions.append(
                UNETConvBlock(in_channels=feature * 2, out_channels=feature),
            )

        # Final part
        self.final_conv = nn.Conv2d(in_channels=feature, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102

        # here we store all skip connections that will reused during upsampling phase
        skip_connections = []

        # phase where we reduce the size of feature maps and increase the number of filters
        # on the scheme (references folder) it's the left side of the U-Net architecture
        for contraction in self.contraction_list:
            x = contraction(x)
            skip_connections.append(x)
            x = self.pooling(x)

        # 'bottom' part of the model
        x = self.bottleneck(x)

        # phase were we increase feature maps size by upsampling and reduce the number of features
        # since a lot of spatial information was lost during contraction phase we use skip_connections
        # in order to restore some it
        # on the scheme it's the right side of the U-Net architecture
        for upsampling, expansion in zip(self.upsampling_list, self.expansions):
            x = upsampling(x)
            skip_connection = skip_connections.pop()

            # tensor with odd size (e.g. 97*97) after max pooling will have
            # even size (48*48), so concatenation might fail. For that reason
            # resize is performed
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = expansion(x)

        return self.final_conv(x)


def _test_architecture() -> None:
    x = torch.randn((1, 1, 388, 388))
    model = UNET(features=UNET_FEATURES, in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    _test_architecture()

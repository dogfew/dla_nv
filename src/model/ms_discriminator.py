import torch
from typing import Callable
import torch.nn as nn
from torch.nn import AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm
from src.model.base_discriminator import BaseDiscriminator


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, norm_fun: Callable = weight_norm):
        super().__init__()
        self.layers = nn.ModuleList([])
        in_channels = 1
        channels_list = [128, 128, 256, 512, 1024, 1024, 1024]
        kernels_list = [15, 41, 41, 41, 41, 41, 5]
        strides_list = [1, 2, 2, 4, 4, 1, 1]
        groups_list = [1, 4, 16, 16, 16, 16, 2]

        for out_channels, kernel_size, stride, groups in zip(
            channels_list, kernels_list, strides_list, groups_list
        ):
            self.layers.append(
                nn.Sequential(
                    norm_fun(
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            groups=groups,
                            padding=(kernel_size // 2),
                        )
                    ),
                    nn.LeakyReLU(0.1),
                )
            )
            in_channels = out_channels
        self.layers.append(
            norm_fun(
                nn.Conv1d(
                    in_channels,
                    1,
                    3,
                    1,
                    padding="same",
                )
            )
        )

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features


class MultiScaleDiscriminator(BaseDiscriminator):
    def __init__(self):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(spectral_norm),
                ScaleDiscriminator(weight_norm),
                ScaleDiscriminator(weight_norm),
            ]
        )
        self.pooling = AvgPool1d(4, 2, padding=2)

    def forward(self, **batch):
        (
            ms_outputs_true,
            ms_outputs_fake,
            ms_features_true,
            ms_features_fake,
        ) = super().forward(**batch)
        return {
            "ms_outputs_true": ms_outputs_true,
            "ms_outputs_fake": ms_outputs_fake,
            "ms_features_true": ms_features_true,
            "ms_features_fake": ms_features_fake,
        }

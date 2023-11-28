import torch
from torch import nn

from src.model.mel_spectrogram import (
    MelSpectrogram,
    MelSpectrogramConfig,
)
from src.model.generator import (
    Generator,
)
from src.model.mp_discriminator import (
    MultiPeriodDiscriminator,
)
from src.model.ms_discriminator import (
    MultiScaleDiscriminator,
)


class HiFiGAN(nn.Module):
    """HiFiGAN"""

    def __init__(self, **kwargs):
        super().__init__()

        self.mel = MelSpectrogram(MelSpectrogramConfig()).cuda()
        self.generator = Generator(
            res_kernel_sizes=[3, 7, 11],
            res_dilation_sizes=[
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            up_init_channels=512,
            up_strides=[8, 8, 2, 2],
            up_kernels=[16, 16, 4, 4],
        )
        self.mp_discriminator = MultiPeriodDiscriminator()
        self.ms_discriminator = MultiScaleDiscriminator()
        #
        # self.mp_discriminator = torch.compile(self.mp_discriminator, mode="reduce-overhead")
        # self.ms_discriminator = torch.compile(self.ms_discriminator, mode="reduce-overhead")

    def forward(self, waves, **kwargs):
        return {
            "mel_true": self.mel(waves),
            "wave_true": waves.unsqueeze(dim=1),
        }

from pathlib import Path

import torch
from torch import nn

current_path = Path(__file__).resolve()


class Mapping_net(nn.Module):
    def __init__(self, latent_dim, layer_num):
        super().__init__()
        self.latent_dim = latent_dim
        self.layer_num = layer_num
        layers = []
        for i in range(self.layer_num):
            layer = nn.Linear(latent_dim, latent_dim)
            nn.init.kaiming_normal_(
                layer.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )
            layers.append(layer)
            if i < layer_num - 1:
                layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class AdaIN(nn.Module):
    def __init__(self, style_dim, channel):
        super().__init__()
        self.affine = nn.Linear(style_dim, channel * 2)
        self.channel = channel
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        # 进行归一化
        x = (x - mean) / (std + 1e-8)  # 为了数值稳定性

        w = self.affine(w)
        gamma = w[:, : self.channel]
        beta = w[:, self.channel :]
        gamma = gamma.reshape(-1, self.channel, 1, 1)
        beta = beta.reshape(-1, self.channel, 1, 1)

        x = gamma * x + beta
        return x


class NoiseInjection(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x: torch.Tensor):
        noise = torch.randn(
            x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
        )
        return x + self.weight * noise


class StyleGANBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, style_dim: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.conv1.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )
        self.noise1 = NoiseInjection(out_channel)
        self.adain1 = AdaIN(style_dim, out_channel)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.conv2.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )
        self.noise2 = NoiseInjection(out_channel)
        self.adain2 = AdaIN(style_dim, out_channel)

    def forward(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.act(x)
        x = self.adain1(x, w1)
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.act(x)
        x = self.adain2(x, w2)
        return x

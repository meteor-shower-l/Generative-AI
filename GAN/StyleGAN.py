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
        w: torch.Tensor,
    ):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.act(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.act(x)
        x = self.adain2(x, w)
        return x


class FirstBlock(nn.Module):
    def __init__(self, style_dim: int, out_channel: int = 512):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, out_channel, 4, 4))
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
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
        w: torch.Tensor,
    ):
        batch_size = w.shape[0]
        x = self.const.repeat(batch_size, 1, 1, 1)
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.act(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.act(x)
        x = self.adain2(x, w)
        return x


class G(nn.Module):
    def __init__(
        self,
        style_dim: int = 512,
        init_channels: int = 512,
        target_resolution: int = 256,
        mapping_layer_num: int = 8,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.maping_net = Mapping_net(style_dim, mapping_layer_num)
        self.FirstBlock = FirstBlock(style_dim=style_dim, out_channel=init_channels)
        self.Blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.style_dim = style_dim
        current_resolution = 4
        self.to_rgb.append(
            nn.Conv2d(
                in_channels=init_channels,
                out_channels=3,
                kernel_size=1,
            )
        )
        while current_resolution < target_resolution:
            in_channel = max((init_channels * 4 // current_resolution), 16)
            out_channel = max((init_channels * 2 // current_resolution), 16)
            self.Blocks.append(
                StyleGANBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    style_dim=style_dim,
                )
            )
            self.to_rgb.append(
                nn.Conv2d(
                    in_channels=out_channel,
                    out_channels=3,
                    kernel_size=1,
                )
            )
            current_resolution *= 2

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        step: int,
        alpha: float,
    ):
        w1 = self.maping_net(z1)
        w2 = self.maping_net(z2)
        total_layers = step + 1
        cutoff = total_layers // 2

        def select_w(layer_idx: int):
            return w1 if layer_idx < cutoff else w2

        x = self.FirstBlock(select_w(0))
        if step == 0:
            return self.to_rgb[0](x)

        for i in range(1, step):
            x = self.Blocks[i - 1](x, select_w(i))

        previous_img = self.to_rgb[step - 1](x)
        previous_img = nn.functional.interpolate(
            previous_img, scale_factor=2, mode="nearest"
        )
        x = self.Blocks[step - 1](x, select_w(step))
        current_img = self.to_rgb[step](x)
        final_img = alpha * current_img + (1.0 - alpha) * previous_img
        return final_img

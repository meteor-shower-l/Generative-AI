from pathlib import Path

import torch
from torch import autograd, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 定义超参数
device = torch.device("cuda")
lr = 1e-4
latent_size = 256
out_res = 128
batch_size = 16 if device.type == "cuda" else 4

current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
dataset_root = project_root / "dataset"
output_dir = current_file.parent / "PGGAN_output"
channels = {
    4: 512,
    8: 512,
    16: 256,
    32: 128,
    64: 64,
    128: 32,
}
# 准备数据
transform = transforms.Compose(
    [
        transforms.Resize(out_res),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
dataset = datasets.CelebA(
    root=str(dataset_root),
    split="all",
    transform=transform,
    download=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
    drop_last=True,
)


# 用于完成channel维度的归一化
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


# 定义一个可以支持上下采样以及完成channel维度初始化的卷积层类
class Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels, PixelNorm=True, upsample=True, downsample=True
    ):
        super().__init__()
        self.PixelNorm = PixelNorm
        self.upsample = upsample
        self.downsample = downsample

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.activate_function = nn.LeakyReLU(0.2, inplace=True)
        self.pixelnorm = PixelNorm()  # type: ignore
        # 如果需要下采样,则通过平均池化完成
        self.pool = nn.AvgPool2d(kernel_size=2) if downsample else None

    def forward(self, x: torch.Tensor):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.activate_function(self.conv1(x))
        if self.use_pixelnorm:
            x = self.pixelnorm(x)
        x = self.activation(self.conv2(x))
        if self.use_pixelnorm:
            x = self.pixelnorm(x)
        if self.downsample:
            x = self.pool(x)  # type: ignore
        return x


class G(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.init_linear = nn.Linear(latent_size, channels[4] * 4 * 4)
        self.init_conv = Conv(channels[4], channels[4], PixelNorm=True)
        self.prog_blocks = nn.ModuleList(
            [
                Conv(channels[4], channels[8], PixelNorm=True, upsample=True),
                Conv(channels[8], channels[16], PixelNorm=True, upsample=True),
                Conv(channels[16], channels[32], PixelNorm=True, upsample=True),
                Conv(channels[32], channels[64], PixelNorm=True, upsample=True),
                Conv(channels[32], channels[64], PixelNorm=True, upsample=True),
                Conv(channels[64], channels[128], PixelNorm=True, upsample=True),
            ]
        )
        self.rgb_blocks = nn.ModuleList(
            [
                nn.Conv2d(channels[4], 3, kernel_size=1),
                nn.Conv2d(channels[8], 3, kernel_size=1),
                nn.Conv2d(channels[16], 3, kernel_size=1),
                nn.Conv2d(channels[32], 3, kernel_size=1),
                nn.Conv2d(channels[64], 3, kernel_size=1),
                nn.Conv2d(channels[128], 3, kernel_size=1),
            ]
        )

    def forward(self, x: torch.Tensor, step: int, alpha: float):
        out = self.init_linear(x)
        out = torch.reshape(out, (-1, channels[4], 4, 4))
        out = self.init_conv(out)

        if step == 0:
            return torch.tanh(self.to_rgbs[0](out))

        for current_step in range(1, step):
            out = self.prog_blocks[current_step - 1](out)

        previous_features_map = out
        out = self.prog_blocks[step - 1](out)
        current_rgb = self.rgb_blocks[step](out)
        previous_rgb = self.rgb_blocks[step - 1](previous_features_map)
        previous_rgb = nn.functional.interpolate(
            previous_rgb, scale_factor=2, mode="nearest"
        )
        # 会返回经过step步放大的图像(rgb)
        return torch.tanh(alpha * current_rgb + (1.0 - alpha) * previous_rgb)


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.prog_blocks = nn.ModuleList(
            [
                Conv(channels[8], channels[4], PixelNorm=False, downsample=True),
                Conv(channels[16], channels[8], PixelNorm=False, downsample=True),
                Conv(channels[32], channels[16], PixelNorm=False, downsample=True),
                Conv(channels[64], channels[32], PixelNorm=False, downsample=True),
                Conv(channels[128], channels[64], PixelNorm=False, downsample=True),
            ]
        )
        self.rbg_blocks = nn.ModuleList(
            [
                nn.Conv2d(3, channels[4], kernel_size=1),
                nn.Conv2d(3, channels[8], kernel_size=1),
                nn.Conv2d(3, channels[16], kernel_size=1),
                nn.Conv2d(3, channels[32], kernel_size=1),
                nn.Conv2d(3, channels[64], kernel_size=1),
                nn.Conv2d(3, channels[128], kernel_size=1),
            ]
        )
        self.final_block = Conv(channels[4], channels[4], PixelNorm=False)
        self.final_linear = nn.Linear(channels[4] * 4 * 4, 1)

    def forward(self, x: torch.Tensor, step: int, alpha: float):
        if step == 0:
            out = self.from_rgbs[0](x)
            out = self.final_block(out)
            out = out.view(out.size(0), -1)
            return self.final_linear(out)

        current_feature_map = self.rbg_blocks[step](x)
        current_feature_map = self.prog_blocks[step - 1](current_feature_map)

        previous_featue_map = nn.functional.avg_pool2d(x, kernel_size=2)
        previous_featue_map = self.rbg_blocks[step - 1](previous_featue_map)

        out = alpha * current_feature_map + (1 - alpha) * previous_featue_map

        for current_step in range(step - 2, -1, -1):
            out = self.prog_blocks[current_step](out)

        out = self.rbg_blocks[0](out)
        out = self.final_block(out)
        out = out.view(out.size(0), -1)
        return self.final_linear(out)

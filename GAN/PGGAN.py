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
    def __init__(self, in_channels, out_channels, PixelNorm, upsample, downsample):
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
        self.pixelnorm = PixelNorm()
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
            x = self.pool(x)
        return x


class G(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

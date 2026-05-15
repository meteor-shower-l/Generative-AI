from pathlib import Path

import torch
from torch import autograd, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 定义超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-4
latent_size = 256
out_res = 128
batch_size = 16
sample_count = 16  # 每次测试时生成器生成图片数量
epoch_per_stage = 1  # 每个分辨率下的训练轮次
d_steps_per_g_step = 1

current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
dataset_root = project_root / "dataset"
output_dir = current_file.parent / "PGGAN_img"
output_dir.mkdir(parents=True, exist_ok=True)
resolutions = [4, 8, 16, 32, 64, 128]
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
    download=False,
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
        self,
        in_channels,
        out_channels,
        use_PixelNorm=True,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self.use_PixelNorm = use_PixelNorm
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
        if self.use_PixelNorm:
            x = self.pixelnorm(x)
        x = self.activate_function(self.conv2(x))
        if self.use_PixelNorm:
            x = self.pixelnorm(x)
        if self.downsample:
            x = self.pool(x)  # type: ignore
        return x


class G(nn.Module):
    def __init__(self, latent_size=latent_size):
        super().__init__()
        self.init_linear = nn.Linear(latent_size, channels[4] * 4 * 4)
        self.init_conv = Conv(channels[4], channels[4], use_PixelNorm=True)
        self.prog_blocks = nn.ModuleList(
            [
                Conv(channels[4], channels[8], use_PixelNorm=True, upsample=True),
                Conv(channels[8], channels[16], use_PixelNorm=True, upsample=True),
                Conv(channels[16], channels[32], use_PixelNorm=True, upsample=True),
                Conv(channels[32], channels[64], use_PixelNorm=True, upsample=True),
                Conv(channels[64], channels[128], use_PixelNorm=True, upsample=True),
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
            return torch.tanh(self.rgb_blocks[0](out))

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
                Conv(channels[8], channels[4], use_PixelNorm=False, downsample=True),
                Conv(channels[16], channels[8], use_PixelNorm=False, downsample=True),
                Conv(channels[32], channels[16], use_PixelNorm=False, downsample=True),
                Conv(channels[64], channels[32], use_PixelNorm=False, downsample=True),
                Conv(channels[128], channels[64], use_PixelNorm=False, downsample=True),
            ]
        )
        self.rgb_blocks = nn.ModuleList(
            [
                nn.Conv2d(3, channels[4], kernel_size=1),
                nn.Conv2d(3, channels[8], kernel_size=1),
                nn.Conv2d(3, channels[16], kernel_size=1),
                nn.Conv2d(3, channels[32], kernel_size=1),
                nn.Conv2d(3, channels[64], kernel_size=1),
                nn.Conv2d(3, channels[128], kernel_size=1),
            ]
        )
        self.final_block = Conv(channels[4], channels[4], use_PixelNorm=False)
        self.final_linear = nn.Linear(channels[4] * 4 * 4, 1)

    def forward(self, x: torch.Tensor, step: int, alpha: float):
        if step == 0:
            out = self.rgb_blocks[0](x)
            out = self.final_block(out)
            out = out.view(out.shape[0], -1)
            return self.final_linear(out)

        current_feature_map = self.rgb_blocks[step](x)
        current_feature_map = self.prog_blocks[step - 1](current_feature_map)

        previous_feature_map = nn.functional.avg_pool2d(x, kernel_size=2)
        previous_feature_map = self.rgb_blocks[step - 1](previous_feature_map)

        out = alpha * current_feature_map + (1 - alpha) * previous_feature_map

        for current_step in range(step - 2, -1, -1):
            out = self.prog_blocks[current_step](out)

        out = self.final_block(out)
        out = out.view(out.shape[0], -1)
        return self.final_linear(out)


# 为了梯度惩罚,即让梯度不超过1,实践中，可以在损失函数中添加一个梯度与1之差的平方
# 下面的函数基于autograd实现了求该惩罚项
def gradient_penalty(
    discriminator: nn.Module,
    real_picture: torch.Tensor,
    fake_picture: torch.Tensor,
    step: int,
    alpha: float,
):
    batch_size_local = real_picture.size(0)
    # 得到真假点连线上的一个点
    # epsilon可以理解为一个随机值
    epsilon = torch.rand(batch_size_local, 1, 1, 1, device=real_picture.device)
    interpolated = epsilon * real_picture + (1 - epsilon) * fake_picture
    interpolated.requires_grad_(True)

    mixed_scores = discriminator(interpolated, step, alpha)
    grad_outputs = torch.ones_like(mixed_scores)
    # 进行求导
    gradients = autograd.grad(
        outputs=mixed_scores,  # 求导的输出
        inputs=interpolated,  # 求导的输入
        # 即求inputs对outputs的导数
        grad_outputs=grad_outputs,  # 对每个输出样本都要进行操作
        create_graph=True,
        retain_graph=True,  # 保留计算图，方便后续使用(不一定必须)
        only_inputs=True,  # 只关心inputs的梯度
    )[0]

    gradients = gradients.view(batch_size_local, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# 用于令生成器生成指定数量的图片并保存
def save_samples(
    generator: nn.Module, step: int, alpha: float, filename, count=sample_count
):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(count, latent_size, device=device)
        samples = generator(z, step, alpha)
        samples = (samples + 1) / 2
        save_image(samples, str(filename), nrow=4)
    generator.train()


generator = G().to(device)
discriminator = D().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
step = 0
for resolution in resolutions:
    print(f"开始训练{resolution}*{resolution}像素的层级")
    for train_times in range(epoch_per_stage):
        print(f"开始训练{train_times + 1}/{epoch_per_stage}轮")
        batch_index = 0
        for real_img, _ in dataloader:
            real_img = real_img.to(device)
            # 将真实图像进行下采样至指定清晰度
            real_img = nn.functional.interpolate(
                real_img,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )
            alpha = min(1.0, batch_index / max(1, (len(dataloader) - 1)))
            total_d_loss = 0
            # 训练d_steps_per_g_step次判别器
            for _ in range(d_steps_per_g_step):
                noise = torch.randn(real_img.shape[0], latent_size, device=device)
                # 由生成器生成伪造图像
                fake_img = generator(
                    noise, step, alpha
                ).detach()  # 截断梯度,确保不训练生成器
                # 得到判别器对真假的图像的判断,在此时传入step与alpha
                real_score = discriminator(real_img, step, alpha)
                fake_score = discriminator(fake_img, step, alpha)
                # 计算梯度惩罚,以限制判别器的梯度
                gradient_p = gradient_penalty(
                    discriminator=discriminator,
                    real_picture=real_img,
                    fake_picture=fake_img,
                    step=step,
                    alpha=alpha,
                )
                # 计算损失函数
                d_loss = fake_score.mean() - real_score.mean() + gradient_p * 10
                total_d_loss += d_loss.item()
                # 反向传播并优化
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            # 训练1次生成器
            noise = torch.randn(real_img.shape[0], latent_size, device=device)
            fake_img = generator(noise, step, alpha)
            g_loss = -discriminator(fake_img, step, alpha).mean()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 完成一个批次的训练,输出训练信息
            average_d_loss = total_d_loss / d_steps_per_g_step
            print(
                f"batch {batch_index}/{len(dataloader)}"
                f"D_loss={average_d_loss:.4f} G_loss={g_loss.item():.4f}"
            )

            # 更新batch_index
            batch_index += 1

    # 每完成一个精确度的训练，生成并保存组图像
    sample_path = output_dir / f"sample_{resolution}x{resolution}.png"
    save_samples(generator, step, 1.0, sample_path)
    # 更新步数
    step += 1

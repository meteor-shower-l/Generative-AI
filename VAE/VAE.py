from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

current_path = Path(__file__).resolve()
data_path = current_path.parents[3] / "dataset"
target_path = current_path.parents[0] / "VAE_img"


# 定义超参数
epoch = 50
lr = 1e-3
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入数据
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
dataset = MNIST(data_path, train=True, transform=transform, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 定义模型
class VAE(nn.Module):
    def __init__(self, input_dim: int, gaussian_dim: int):
        super().__init__()
        # 编码器定义
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(in_features=256, out_features=gaussian_dim)
        self.log_sigma = nn.Linear(in_features=256, out_features=gaussian_dim)

        # 解码器定义
        self.decoder = nn.Sequential(
            nn.Linear(in_features=gaussian_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)

        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        sigma = torch.exp(log_sigma * 0.5)
        e = torch.randn_like(sigma, device=device)
        h_sample = e * sigma + mu

        result = self.decoder(h_sample)

        return result, mu, log_sigma

    def predict(self, x: torch.Tensor):
        return self.decoder(x)


# 实例化网络
net = VAE(input_dim=784, gaussian_dim=20).to(device)
# 定义损失函数与优化器
loss_fn = nn.MSELoss(reduction="sum").to(device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
# 开始训练
for train_times in range(epoch):
    total_loss = 0
    total_num = len(dataset)
    for features, _ in dataloader:
        features = features.to(device)
        features = torch.reshape(features, (-1, 784))

        result, mu, log_sigma = net(features)
        # 似然损失
        loss_likelihood = loss_fn(features, result)
        # KL损失
        loss_KL = torch.pow(mu, 2) + torch.exp(log_sigma) - log_sigma - 1
        # 总损失
        loss = loss_likelihood + 0.5 * torch.sum(loss_KL)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_average = total_loss / total_num
    print(f"完成第{train_times + 1}次训练,平均损失为{loss_average}")
    if (train_times + 1) % 10 == 0:
        with torch.no_grad():
            test = torch.randn((10, 20)).to(device)
            result = net.predict(test).detach().cpu()
            result = torch.reshape(result, (-1, 1, 28, 28))
            name = f"result_{train_times + 1}.png"
            torchvision.utils.save_image(result, target_path / name)

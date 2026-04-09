"""
若生成器损失稳定在log(2)=0.69,判别器损失稳定在2*log(2)=1.38左右
代表两者达到纳什均衡状态

需要注意的是两者达到纳什均衡状态无法说明两者均达到较好的性能，只能说明两者平衡
两者可能都很强，也可能都很弱
若希望判断两者性能，应当使用验证机另外判断

在目前参数下，可以达到还不错的生成效果
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def show_outputs(outputs):
    imgs = outputs.detach().cpu().numpy()
    imgs = imgs.reshape(-1, 28, 28)
    imgs = (imgs + 1) / 2  # 从[-1,1]转换到[0,1]

    # 显示10张图片
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# 获取数据并定义dataloader
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)
data = MNIST(
    root="dataset",
    train=True,
    download=False,
    transform=transform,
)
True_loader = DataLoader(dataset=data, batch_size=64, shuffle=True)

# 定义超参数
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 200
lr = 1e-4

# 定义网络
# 定义判别器网络
D = nn.Sequential(
    nn.Linear(784, 512),
    nn.Dropout(p=0.3),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.Dropout(p=0.3),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Dropout(p=0.3),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)
D = D.to(device)

# 定义生成器网络
G = nn.Sequential(
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 784),
    nn.Tanh(),
)
G = G.to(device)

# 定义优化器
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
# 定义损失函数
loss_function = torch.nn.BCELoss().to(device)

# 开始训练
for times in range(epoch):
    D_total_loss = 0
    G_total_loss = 0
    total_num = 0
    D.train()
    G.train()
    for features, temp in True_loader:
        sample_num = features.shape[0]
        true_lables = torch.ones(sample_num, 1, device=device)
        total_num += 1
        # 转移至GPU
        true_features = features.reshape(-1, 784).to(device)
        true_lables = true_lables.to(device)
        # 判别器输出
        outputs = D(true_features)
        # 计算损失
        true_loss = loss_function(outputs, true_lables)
        # 生成伪造数据
        false_features = torch.normal(
            mean=0, std=1, size=(sample_num, 128), device=device
        )
        false_labels = torch.zeros(sample_num, 1, device=device)
        outputs_of_G = G(false_features)
        Judge_of_D = D(outputs_of_G.detach())

        false_loss = loss_function(Judge_of_D, false_labels)
        D_loss = false_loss + true_loss
        D_total_loss += D_loss.item()
        # 清空梯度,反向传播,执行优化
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        G_loss = loss_function(D(outputs_of_G), true_lables)
        G_total_loss += G_loss
        # 清空梯度,反向传播,执行优化
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
    average_D_loss = D_total_loss / total_num
    average_G_loss = G_total_loss / total_num
    print(
        f"第{times + 1}轮训练后，判别器平均损失为:{average_D_loss},生成器平均损失为:{average_G_loss}"
    )
features = torch.normal(mean=0, std=1, size=(10, 128), device=device)
outputs = G(features)
show_outputs(outputs)

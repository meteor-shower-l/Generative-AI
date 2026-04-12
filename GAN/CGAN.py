# Conditional GAN 条件GAN


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def save_result_to_image(result, filename="generated_images.png"):
    """
    保存生成器生成的图片
    result: 生成器输出的张量，形状应为(num_images, 784)或(num_images, 1, 28, 28)
    filename: 保存的文件名
    """
    import torchvision

    # 调整形状为(batch_size, 1, 28, 28)格式
    if result.dim() == 2 and result.shape[1] == 784:
        result = result.view(-1, 1, 28, 28)

    # 反归一化: [-1, 1] -> [0, 1]
    result = (result + 1) / 2

    # 创建网格并保存
    grid = torchvision.utils.make_grid(result, nrow=5, padding=2, normalize=False)
    torchvision.utils.save_image(grid, filename)


# 定义超参数
# 学习率
lr = 1e-4
# 训练轮数
train_epoch = 50
# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"


# 定义D(判别器)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear_1_1 = nn.Linear(784, 1024)
        self.Linear_1_2 = nn.Linear(10, 1024)
        self.Linear_2 = nn.Linear(2048, 512)
        self.Linear_3 = nn.Linear(512, 256)
        self.Linear_4 = nn.Linear(256, 64)
        self.Linear_5 = nn.Linear(64, 1)

    def forward(self, x, y):
        features = F.leaky_relu(self.Linear_1_1(x), 0.2)
        labels = F.leaky_relu(self.Linear_1_2(y), 0.2)
        features = torch.cat([features, labels], 1)
        features = F.leaky_relu(((self.Linear_2(features))), 0.2)
        features = F.leaky_relu(((self.Linear_3(features))), 0.2)
        features = F.leaky_relu(((self.Linear_4(features))), 0.2)
        features = F.sigmoid(self.Linear_5(features))

        return features


# 定义G(生成器)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear_1_1 = nn.Linear(128, 256)
        self.bn_1_1 = nn.BatchNorm1d(256)
        self.Linear_1_2 = nn.Linear(10, 256)
        self.bn_1_2 = nn.BatchNorm1d(256)
        self.Linear_2 = nn.Linear(512, 512)
        self.bn_2 = nn.BatchNorm1d(512)
        self.Linear_3 = nn.Linear(512, 1024)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.Linear_4 = nn.Linear(1024, 784)

    def forward(self, x, y):
        features = F.leaky_relu(self.bn_1_1(self.Linear_1_1(x)), 0.2)
        labels = F.leaky_relu(self.bn_1_2(self.Linear_1_2(y)), 0.2)
        features = torch.cat([features, labels], 1)
        features = F.leaky_relu(self.bn_2(self.Linear_2(features)), 0.2)
        features = F.leaky_relu(self.bn_3(self.Linear_3(features)), 0.2)
        features = F.tanh(self.Linear_4(features))

        return features


# 实例化判别器、生成器对象
D = Discriminator()
D = D.to(device)
G = Generator()
G = G.to(device)

# 定义优化器
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
# 定义损失函数
loss_function = torch.nn.BCELoss().to(device)

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

# 定义标准数据，用于测试使用
fixed_false_features = torch.normal(mean=0, std=1, size=(10, 128), device=device)
standard_label = torch.eye(10, device=device)
# 训练
for times in range(train_epoch):
    D_total_loss = 0
    G_total_loss = 0
    total_num = 0
    for features, labels in True_loader:
        total_num += 1
        # 准备数据(标签、正态分布)
        features = features.reshape(-1, 784).to(device)
        labels = labels.to(device)
        sample_num = features.shape[0]
        y_true = torch.ones(size=(sample_num, 1), device=device)
        y_false = torch.zeros(size=(sample_num, 1), device=device)
        y_labels = torch.zeros(sample_num, 10, device=device)
        y_labels.scatter_(1, labels.view(sample_num, 1), 1)
        false_features = torch.normal(
            mean=0, std=1, size=(sample_num, 128), device=device
        )

        # 固定G，训练D
        D.train()
        G.eval()
        # 获取D的真样本判断损失
        D_result_true = D(features, y_labels)
        D_True_Loss = loss_function(D_result_true, y_true)
        # 获取D的假样本判断损失
        G_result = G(false_features, y_labels)
        D_result_false = D(G_result.detach(), y_labels)
        D_False_Loss = loss_function(D_result_false, y_false)
        D_Loss = D_True_Loss + D_False_Loss
        D_total_loss += D_Loss.item()
        # 得到D的总损失并反向传播
        D.zero_grad()
        D_Loss.backward()
        D_optimizer.step()

        # 固定D，训练G
        D.eval()
        G.train()
        G_result = G(false_features, y_labels)
        D_judge = D(G_result, y_labels)
        G_Loss = loss_function(D_judge, y_true)
        G_total_loss += G_Loss.item()
        G.zero_grad()
        G_Loss.backward()
        G_optimizer.step()

    # 输出训练信息
    average_D_loss = D_total_loss / total_num
    average_G_loss = G_total_loss / total_num
    print(
        f"完成第{times + 1}轮训练,判别器损失:{average_D_loss},生成器损失:{average_G_loss}"
    )
    if (times + 1) % 10 == 0:
        result = G(fixed_false_features, standard_label)
        save_result_to_image(result, f"generated_epoch_{times + 1}.png")

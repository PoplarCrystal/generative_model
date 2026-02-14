import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """生成器：从噪声生成图像（与 GAN 相同结构）"""
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(noise_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.linear4 = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, z):
        h = F.leaky_relu(self.bn1(self.linear1(z)), 0.2)
        h = F.leaky_relu(self.bn2(self.linear2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.linear3(h)), 0.2)
        x = torch.tanh(self.linear4(h))  # 输出范围 [-1, 1]
        return x


class Critic(nn.Module):
    """Critic（评论家）：输出真实/生成样本的 Wasserstein 距离相关分数，无 sigmoid"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # 最后一层无激活，输出任意实数（Wasserstein 距离的近似）

    def forward(self, x):
        h = F.leaky_relu(self.linear1(x), 0.2)
        h = F.leaky_relu(self.linear2(h), 0.2)
        h = F.leaky_relu(self.linear3(h), 0.2)
        out = self.linear4(h)  # 无 sigmoid，WGAN 需要未裁剪的分数
        return out

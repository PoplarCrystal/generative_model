import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """生成器：从噪声生成图像"""
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super().__init__()
        # 参考实现的结构：noise_dim -> hidden_dim -> hidden_dim*2 -> hidden_dim*4 -> output_dim
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


class Discriminator(nn.Module):
    """判别器：判断输入是真实图像还是生成图像"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 参考实现的结构：input_dim -> hidden_dim*4 -> hidden_dim*2 -> hidden_dim -> 1
        self.linear1 = nn.Linear(input_dim, hidden_dim * 4)
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.dropout1(F.leaky_relu(self.linear1(x), 0.2))
        h = self.dropout2(F.leaky_relu(self.linear2(h), 0.2))
        h = self.dropout3(F.leaky_relu(self.linear3(h), 0.2))
        out = torch.sigmoid(self.linear4(h))
        return out


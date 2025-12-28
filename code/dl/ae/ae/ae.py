import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        z = self.linear2(h)  # 直接输出潜在向量，不需要概率分布
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        x_hat = F.sigmoid(self.linear2(h))
        return x_hat


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)  # 编码：x -> z
        x_hat = self.decoder(z)  # 解码：z -> x_hat
        return x_hat


def loss_function(x_hat, x):
    # AE 只有重构损失，没有 KL 散度
    return F.mse_loss(x_hat, x, reduction='mean')


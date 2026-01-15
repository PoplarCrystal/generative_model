import math
import torch
from torch import nn


def _pos_encoding(time_idx, output_dim, device='cpu'):
    """位置编码的单个时间步实现"""
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


def timestep_embedding(timesteps, dim, device='cpu'):
    """位置编码：将时间步编码为向量"""
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], dim, device)
    return v


class ConvBlock(nn.Module):
    """带时间嵌入的卷积块"""
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, emb):
        N, C, _, _ = x.shape
        emb = self.mlp(emb)
        emb = emb.view(N, C, 1, 1)
        y = self.convs(x + emb)
        return y


class UNetModel(nn.Module):
    """
    UNet 网络：用于预测噪声
    
    :param in_channels: 输入通道数
    :param model_channels: 基础通道数
    :param time_embed_dim: 时间嵌入维度
    """
    def __init__(self, in_channels=1, model_channels=64, time_embed_dim=100):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_channels, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_channels, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x H x W] Tensor of outputs.
        """
        emb = timestep_embedding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, emb)
        x = self.maxpool(x1)
        x2 = self.down2(x, emb)
        x = self.maxpool(x2)

        x = self.bot1(x, emb)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, emb)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, emb)
        x = self.out(x)
        return x


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


def pos_encoding(timesteps, output_dim, device='cpu'):
    """批量位置编码"""
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
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

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class LatentAdapter(nn.Module):
    """适配层：将 1D 潜在向量转换为 2D 特征图供 UNet 使用"""
    def __init__(self, latent_dim, feature_size=8, feature_channels=4):
        super().__init__()
        # 将潜在向量 reshape 成小的 2D 特征图
        # feature_size * feature_size * feature_channels = latent_dim
        self.feature_size = feature_size
        self.feature_channels = feature_channels
        self.linear = nn.Linear(latent_dim, feature_size * feature_size * feature_channels)
    
    def forward(self, z):
        # z: [B, latent_dim] -> [B, feature_channels, feature_size, feature_size]
        B = z.shape[0]
        h = self.linear(z)
        h = h.view(B, self.feature_channels, self.feature_size, self.feature_size)
        return h


class UNetCond(nn.Module):
    """条件 U-Net 架构用于预测噪声，支持标签条件
    在 LDM 中，UNet 在潜在空间工作
    """
    def __init__(self, latent_dim, feature_size=8, feature_channels=4, 
                 time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        # 适配层：1D 潜在向量 -> 2D 特征图
        self.adapter = LatentAdapter(latent_dim, feature_size, feature_channels)
        
        # UNet 在特征图上工作
        self.down1 = ConvBlock(feature_channels, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, feature_channels, 1)
        
        # 输出适配层：2D 特征图 -> 1D 潜在向量
        # 使用 flatten 而不是 AdaptiveAvgPool2d，保留空间信息
        self.output_adapter = nn.Sequential(
            nn.Flatten(),  # [B, feature_channels, H, W] -> [B, feature_channels*H*W]
            nn.Linear(feature_channels * feature_size * feature_size, latent_dim)  # -> [B, latent_dim]
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 标签嵌入层（用于条件生成）
        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)
        else:
            self.label_emb = None

    def forward(self, z, timesteps, labels=None):
        """
        Args:
            z: [B, latent_dim] 潜在向量
            timesteps: [B] 时间步
            labels: [B] 条件标签（可选）
        Returns:
            eps: [B, latent_dim] 预测的噪声
        """
        # 将潜在向量转换为 2D 特征图
        x = self.adapter(z)  # [B, latent_dim] -> [B, feature_channels, H, W]
        
        v = pos_encoding(timesteps, self.time_embed_dim, z.device)
        
        # 如果有标签，将标签嵌入加到时间嵌入中
        if labels is not None and self.label_emb is not None:
            v = v + self.label_emb(labels)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)  # [B, feature_channels, H, W]
        
        # 将 2D 特征图转换回 1D 潜在向量
        eps = self.output_adapter(x)  # [B, feature_channels, H, W] -> [B, latent_dim]
        return eps


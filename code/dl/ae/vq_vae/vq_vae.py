import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块，用于增强模型表达能力"""
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAE(nn.Module):
    """VQ-VAE 模型（CNN 架构，参考 DL-Demos）"""
    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        # Encoder: 使用卷积层下采样
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),  # 7x7 -> 7x7
            ResidualBlock(dim),
            ResidualBlock(dim)
        )
        
        # Codebook：可学习的嵌入向量表
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        # 初始化 codebook：参考原始论文
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        
        # Decoder: 使用转置卷积上采样
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1)  # 14x14 -> 28x28
        )
        
        self.n_downsample = 2  # 下采样次数

    def forward(self, x):
        """
        训练时的前向传播
        返回: (x_hat, ze, zq)
        - x_hat: 重构图像
        - ze: 编码器输出的连续潜在表示（用于计算损失）
        - zq: 量化后的向量（用于计算损失）
        """
        # encode: x -> ze (连续潜在表示)
        ze = self.encoder(x)  # (N, C, H, W)
        
        # 向量量化
        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)  # (N, H, W)
        
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)  # (N, C, H, W)
        
        # stop gradient: 使用 straight-through estimator
        decoder_input = ze + (zq - ze).detach()
        
        # decode: zq -> x_hat
        x_hat = self.decoder(decoder_input)
        
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        """
        编码为离散的 codebook 索引（推理时使用）
        用途：获取图像的离散表示，可用于后续的自回归模型（如 PixelCNN）
        返回: (N, H, W) 的索引张量
        """
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data
        
        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        """
        从 codebook 索引解码（推理时使用）
        用途：从离散索引生成图像，可用于生成新图像
        discrete_latent: (N, H, W) 的索引张量
        """
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    def get_latent_HW(self, input_shape):
        """
        获取潜在空间的空间维度
        input_shape: (C, H, W)
        返回: (H', W') 潜在空间的高度和宽度
        """
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)


def loss_function(x_hat, x, ze, zq, embedding_loss_weight=1.0, commitment_loss_weight=0.25):
    """
    VQ-VAE 损失函数（与参考代码一致）
    - 重构损失：衡量重构质量
    - Embedding 损失：将 codebook 向量拉向 ze（更新 codebook）
    - Commitment 损失：将 ze 拉向 codebook 向量（鼓励 encoder 输出接近 codebook）
    
    参数:
        embedding_loss_weight: embedding 损失的权重（默认 1.0）
        commitment_loss_weight: commitment 损失的权重（默认 0.25）
    """
    # 重构损失
    recon_loss = F.mse_loss(x_hat, x)
    
    # Embedding 损失：将 codebook 向量拉向 ze（更新 codebook）
    embedding_loss = F.mse_loss(ze.detach(), zq)
    
    # Commitment 损失：将 ze 拉向 codebook 向量（鼓励 encoder 输出接近 codebook）
    commit_loss = F.mse_loss(ze, zq.detach())
    
    total_loss = recon_loss + embedding_loss_weight * embedding_loss + commitment_loss_weight * commit_loss
    
    return total_loss, recon_loss, embedding_loss, commit_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """VAE 编码器：将图像编码到潜在空间"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, C*H*W]
        B = x.shape[0]
        x_flat = x.view(B, -1)
        h = F.relu(self.linear(x_flat))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    """VAE 解码器：将潜在向量解码回图像"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = torch.sigmoid(h)
        return x_hat


class VAE(nn.Module):
    """VAE 模型：用于图像和潜在空间之间的转换"""
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """编码：图像 -> 潜在向量"""
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        return z, mu, sigma
    
    def decode(self, z):
        """解码：潜在向量 -> 图像"""
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        """前向传播：编码 -> 解码"""
        z, mu, sigma = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma):
        """重参数化技巧"""
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


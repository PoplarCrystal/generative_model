import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.linear(x))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma): # 重参数化
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


def loss_function(x_hat, x, mu, sigma):
    batch_size = x.shape[0]
    L1 = F.mse_loss(x_hat, x, reduction='sum')
    L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
    return (L1 + L2) / batch_size


# ========== VAE_CNN (卷积版本) ==========

class Encoder_CNN(nn.Module):
    """简化的 CNN Encoder：更稳定的架构"""
    def __init__(self, input_channel, hidden_channel, latent_channel):
        super().__init__()
        # 28x28 -> 14x14
        self.conv1 = nn.Conv2d(input_channel, hidden_channel, kernel_size=4, stride=2, padding=1)
        # 14x14 -> 7x7
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel * 2, kernel_size=4, stride=2, padding=1)
        # 输出 mu 和 logvar
        self.conv_mu = nn.Conv2d(hidden_channel * 2, latent_channel, kernel_size=7, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(hidden_channel * 2, latent_channel, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        h = F.relu(self.conv1(x))  # 28x28 -> 14x14
        h = F.relu(self.conv2(h))   # 14x14 -> 7x7
        mu = self.conv_mu(h)        # 7x7 -> 1x1
        logvar = self.conv_logvar(h)  # 7x7 -> 1x1
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder_CNN(nn.Module):
    """简化的 CNN Decoder：更稳定的架构"""
    def __init__(self, latent_channel, hidden_channel, output_channel):
        super().__init__()
        # 1x1 -> 7x7
        self.conv1 = nn.ConvTranspose2d(latent_channel, hidden_channel * 2, kernel_size=7, stride=1, padding=0)
        # 7x7 -> 14x14
        self.conv2 = nn.ConvTranspose2d(hidden_channel * 2, hidden_channel, kernel_size=4, stride=2, padding=1)
        # 14x14 -> 28x28
        self.conv3 = nn.ConvTranspose2d(hidden_channel, output_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.conv1(z))   # 1x1 -> 7x7
        h = F.relu(self.conv2(h))     # 7x7 -> 14x14
        x_hat = torch.sigmoid(self.conv3(h))  # 14x14 -> 28x28
        return x_hat


class VAE_CNN(nn.Module):
    """VAE_CNN: 使用卷积层的 VAE，保持图像的空间结构"""
    def __init__(self, input_channel, output_channel, hidden_channel, latent_channel):
        super().__init__()
        self.encoder = Encoder_CNN(input_channel, hidden_channel, latent_channel)
        self.decoder = Decoder_CNN(latent_channel, hidden_channel, output_channel)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma): # 重参数化
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z

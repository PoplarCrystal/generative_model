import torch
import torch.nn as nn
import torch.nn.functional as F


# 原始标签：lbls = tensor([3, 7, 1, ...])
# 转换为 one-hot：condition = tensor([
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 数字3
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 数字7
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 数字1
#     ...
# ])
def onehot(x, max_dim):
    # 方法1：使用 F.one_hot（最简单，推荐）
    return F.one_hot(x, num_classes=max_dim).float()


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


class CVAE(nn.Module):
    """CVAE: 条件在编码时与输入拼接"""
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim + condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
    
    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        z = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma): # 重参数化
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


class CVAE1(nn.Module):
    """CVAE1: 条件只在解码时使用，编码时不用条件"""
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
    
    def forward(self, x, c):
        # 编码时不使用条件
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        # 解码时使用条件
        z = torch.cat((z, c), dim=1)
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

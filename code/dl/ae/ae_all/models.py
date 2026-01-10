"""
AE, VAE, CVAE模型定义
所有模型使用latent_dim=2用于2D可视化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== AE (Autoencoder) ==========
class AE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        z = self.linear2(h)  # 直接输出潜在向量
        return z


class AE_Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128, output_dim=784):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        x_hat = torch.sigmoid(self.linear2(h))
        return x_hat


class AE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.encoder = AE_Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = AE_Decoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ========== VAE (Variational Autoencoder) ==========
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=2):
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


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128, output_dim=784):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        x_hat = torch.sigmoid(self.linear2(h))
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.encoder = VAE_Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


# ========== CVAE (Conditional Variational Autoencoder) ==========
class ConditionEmbedding(nn.Module):
    """条件嵌入层：将类别标签转换为embedding向量"""
    def __init__(self, n_classes=10, embed_dim=5):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim, max_norm=True)
        self.embed_dim = embed_dim
    
    def forward(self, labels):
        # 确保labels是一维的
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        return self.embedding(labels)


class CVAE_Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim=2):
        super().__init__()
        # 输入是x和c的拼接
        self.linear = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        # 拼接x和c
        x_c = torch.cat([x, c], dim=1)
        h = F.relu(self.linear(x_c))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim=2, condition_dim=5, hidden_dim=128, output_dim=784):
        super().__init__()
        # 输入是z和c的拼接
        self.linear1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, c):
        # 拼接z和c
        z_c = torch.cat([z, c], dim=1)
        h = F.relu(self.linear1(z_c))
        x_hat = torch.sigmoid(self.linear2(h))
        return x_hat


class CVAE(nn.Module):
    def __init__(self, input_dim=784, n_classes=10, embed_dim=5, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.condition_embedding = ConditionEmbedding(n_classes, embed_dim)
        self.condition_dim = embed_dim  # 使用embedding维度
        self.encoder = CVAE_Encoder(input_dim, self.condition_dim, hidden_dim, latent_dim)
        self.decoder = CVAE_Decoder(latent_dim, self.condition_dim, hidden_dim, input_dim)
    
    def forward(self, x, labels):
        # 将标签转换为embedding
        c = self.condition_embedding(labels)
        mu, sigma = self.encoder(x, c)
        z = self._reparameterize(mu, sigma)
        x_hat = self.decoder(z, c)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


# ========== 损失函数 ==========
def ae_loss_function(x_hat, x):
    """AE损失函数：只有重构损失"""
    return F.mse_loss(x_hat, x, reduction='mean')


def vae_loss_function(x_hat, x, mu, sigma):
    """VAE损失函数：重构损失 + KL散度"""
    batch_size = x.shape[0]
    # 重构损失
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL散度: KL(q(z|x) || N(0, I))
    kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(sigma + 1e-8) - mu ** 2 - sigma ** 2)
    return (recon_loss + kl_loss) / batch_size


def cvae_loss_function(x_hat, x, mu, sigma):
    """CVAE损失函数：重构损失 + KL散度（与VAE相同）"""
    return vae_loss_function(x_hat, x, mu, sigma)


# ========== CVAE_prior (Conditional VAE with Prior Network) ==========
class PriorNetwork(nn.Module):
    """先验网络：学习条件先验分布 p(z|c) = N(μ_prior(c), σ_prior(c))"""
    def __init__(self, condition_dim, hidden_dim, latent_dim=2):
        super().__init__()
        self.linear = nn.Linear(condition_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, c):
        h = F.relu(self.linear(c))
        mu_prior = self.linear_mu(h)
        logvar_prior = self.linear_logvar(h)
        sigma_prior = torch.exp(0.5 * logvar_prior)
        return mu_prior, sigma_prior


class CVAE_prior(nn.Module):
    """CVAE with Prior Network: p(z|c) = N(μ_prior(c), σ_prior(c))"""
    def __init__(self, input_dim=784, n_classes=10, embed_dim=5, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.condition_embedding = ConditionEmbedding(n_classes, embed_dim)
        self.condition_dim = embed_dim  # 使用embedding维度
        self.encoder = CVAE_Encoder(input_dim, self.condition_dim, hidden_dim, latent_dim)
        self.prior = PriorNetwork(self.condition_dim, hidden_dim, latent_dim)  # 先验网络
        self.decoder = CVAE_Decoder(latent_dim, self.condition_dim, hidden_dim, input_dim)
    
    def forward(self, x, labels):
        # 将标签转换为embedding
        c = self.condition_embedding(labels)
        # 编码: q(z|x, c)
        mu_q, sigma_q = self.encoder(x, c)
        z = self._reparameterize(mu_q, sigma_q)
        
        # 先验: p(z|c)
        mu_prior, sigma_prior = self.prior(c)
        
        # 解码: p(x|z, c)
        x_hat = self.decoder(z, c)
        
        return x_hat, mu_q, sigma_q, mu_prior, sigma_prior
    
    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


def cvae_prior_loss_function(x_hat, x, mu_q, sigma_q, mu_prior, sigma_prior):
    """CVAE_prior损失函数：重构损失 + KL散度(q(z|x,c) || p(z|c))"""
    batch_size = x.shape[0]
    
    # 重构损失
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度: KL(q(z|x,c) || p(z|c))
    logvar_q = 2 * torch.log(sigma_q + 1e-8)
    logvar_prior = 2 * torch.log(sigma_prior + 1e-8)
    
    kl_loss = 0.5 * torch.sum(
        logvar_prior - logvar_q + 
        (sigma_q ** 2 + (mu_q - mu_prior) ** 2) / (2.0 * (sigma_prior ** 2 + 1e-8)) - 
        1.0
    )
    
    return (recon_loss + kl_loss) / batch_size


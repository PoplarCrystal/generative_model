"""
CVAE (Conditional Variational Autoencoder) 实现

本文件实现了多种CVAE变体，考虑了条件c和潜在变量z之间的关系：

1. CVAE: 条件在编码时与输入拼接，先验 p(z) = N(0, I)
   - 编码: q(z|x, c)
   - 先验: p(z) = N(0, I) - 固定标准正态分布
   - 解码: p(x|z, c)
   - KL项: KL(q(z|x,c) || N(0, I))

2. CVAE1: 条件只在解码时使用，先验 p(z) = N(0, I)
   - 编码: q(z|x) - 不使用条件
   - 先验: p(z) = N(0, I) - 固定标准正态分布
   - 解码: p(x|z, c)
   - KL项: KL(q(z|x) || N(0, I))

3. CVAE_prior: 条件在编码时使用，有Prior网络
   - 编码: q(z|x, c)
   - 先验: p(z|c) = N(μ_prior(c), σ_prior(c)) - 由Prior网络学习
   - 解码: p(x|z, c)
   - KL项: KL(q(z|x,c) || p(z|c))

4. CVAE1_prior: 条件只在解码时使用，但有Prior网络
   - 编码: q(z|x) - 不使用条件
   - 先验: p(z|c) = N(μ_prior(c), σ_prior(c)) - 由Prior网络学习
   - 解码: p(x|z, c)
   - KL项: KL(q(z|x) || p(z|c))

理论背景：
CVAE的目标函数是ELBO，关键区别在于：
1. 编码器是否使用条件：q(z|x,c) vs q(z|x)
2. 先验分布是否依赖于条件：p(z) vs p(z|c)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotEncoder(nn.Module):
    """One-hot编码器模块"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self, labels):
        return F.one_hot(labels, num_classes=self.n_classes).float()


class ConditionEncoder(nn.Module):
    """
    条件编码器：支持one-hot和embedding两种编码方式
    """
    def __init__(self, n_classes, condition_encoding_type='onehot', embed_dim=None):
        """
        Args:
            n_classes: 类别数量（如MNIST是10）
            condition_encoding_type: 'onehot' 或 'embedding'
            embed_dim: embedding的维度，仅在condition_encoding_type='embedding'时需要
        """
        super().__init__()
        self.n_classes = n_classes
        self.condition_encoding_type = condition_encoding_type
        
        if condition_encoding_type == 'embedding':
            if embed_dim is None:
                raise ValueError("使用embedding编码时，必须提供embed_dim参数")
            self.encoder = nn.Embedding(
                num_embeddings=n_classes,
                embedding_dim=embed_dim,
                max_norm=True
            )
            self.condition_dim = embed_dim  # embedding时，条件维度等于embed_dim
        elif condition_encoding_type == 'onehot':
            self.encoder = OneHotEncoder(n_classes)
            self.condition_dim = n_classes  # onehot时，条件维度等于n_classes
        else:
            raise ValueError(f"不支持的condition_encoding_type: {condition_encoding_type}，请选择 'onehot' 或 'embedding'")
    
    def forward(self, labels):
        """
        Args:
            labels: 标签张量，shape为 (batch_size,) 或 (batch_size, 1)
        
        Returns:
            编码后的条件张量，shape为 (batch_size, condition_dim)
        """
        # 确保labels是一维的
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        return self.encoder(labels)


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
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim, 
                 condition_encoder=None):
        """
        Args:
            condition_encoder: ConditionEncoder实例，如果为None则假设输入c已经是编码后的
        """
        super().__init__()
        self.condition_encoder = condition_encoder
        self.encoder = Encoder(input_dim + condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
    
    def forward(self, x, c_or_labels):
        """
        Args:
            x: 输入数据
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 如果提供了condition_encoder，则c_or_labels是标签，需要编码
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels
        
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
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim,
                 condition_encoder=None):
        """
        Args:
            condition_encoder: ConditionEncoder实例，如果为None则假设输入c已经是编码后的
        """
        super().__init__()
        self.condition_encoder = condition_encoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
    
    def forward(self, x, c_or_labels):
        """
        Args:
            x: 输入数据
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 如果提供了condition_encoder，则c_or_labels是标签，需要编码
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels
        
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


class PriorNetwork(nn.Module):
    """先验网络：学习条件先验分布 p(z|c) = N(μ_prior(c), σ_prior(c))"""
    def __init__(self, condition_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(condition_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, c):
        """
        输入条件c，输出先验分布的参数
        返回: (mu_prior, sigma_prior) 其中 p(z|c) = N(mu_prior, sigma_prior^2)
        """
        h = F.relu(self.linear(c))
        mu_prior = self.linear_mu(h)
        logvar_prior = self.linear_logvar(h)
        sigma_prior = torch.exp(0.5 * logvar_prior)
        return mu_prior, sigma_prior


class CVAE_prior(nn.Module):
    """
    CVAE with Prior Network: p(z|c) = N(μ_prior(c), σ_prior(c))
    假设潜在变量z和条件c是非独立的，先验分布依赖于条件c
    需要Prior网络来学习 p(z|c)
    """
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim,
                 condition_encoder=None):
        """
        Args:
            condition_encoder: ConditionEncoder实例，如果为None则假设输入c已经是编码后的
        """
        super().__init__()
        self.condition_encoder = condition_encoder
        self.encoder = Encoder(input_dim + condition_dim, hidden_dim, latent_dim)
        self.prior = PriorNetwork(condition_dim, hidden_dim, latent_dim)  # 先验网络
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x, c_or_labels):
        """
        Args:
            x: 输入数据
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 如果提供了condition_encoder，则c_or_labels是标签，需要编码
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels
        
        # 编码: q(z|x, c)
        x_c = torch.cat((x, c), dim=1)
        mu_q, sigma_q = self.encoder(x_c)
        z = self._reparameterize(mu_q, sigma_q)
        
        # 先验: p(z|c)
        mu_prior, sigma_prior = self.prior(c)
        
        # 解码: p(x|z, c)
        z_c = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z_c)
        
        return x_hat, mu_q, sigma_q, mu_prior, sigma_prior
    
    def sample_prior(self, c_or_labels):
        """
        从条件先验分布采样：p(z|c) = N(μ_prior(c), σ_prior(c))
        Args:
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 如果提供了condition_encoder，则c_or_labels是标签，需要编码
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels
        
        mu_prior, sigma_prior = self.prior(c)
        z = self._reparameterize(mu_prior, sigma_prior)
        return z
    
    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


class CVAE1_prior(nn.Module):
    """
    CVAE1 with Prior Network: 条件只在解码时使用，编码时不用条件，但有Prior网络
    - 编码: q(z|x) - 编码时不使用条件
    - 先验: p(z|c) = N(μ_prior(c), σ_prior(c)) - 由Prior网络学习
    - 解码: p(x|z, c) - 解码时使用条件
    
    ELBO = E_q(z|x)[log p(x|z,c)] - KL(q(z|x) || p(z|c))
    """
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim,
                 condition_encoder=None):
        """
        Args:
            condition_encoder: ConditionEncoder实例，如果为None则假设输入c已经是编码后的
        """
        super().__init__()
        self.condition_encoder = condition_encoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)  # 编码时不使用条件
        self.prior = PriorNetwork(condition_dim, hidden_dim, latent_dim)  # 先验网络
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x, c_or_labels):
        """
        Args:
            x: 输入数据
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 如果提供了condition_encoder，则c_or_labels是标签，需要编码
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels
        
        # 编码: q(z|x) - 不使用条件
        mu_q, sigma_q = self.encoder(x)
        z = self._reparameterize(mu_q, sigma_q)
        
        # 先验: p(z|c)
        mu_prior, sigma_prior = self.prior(c)
        
        # 解码: p(x|z, c) - 使用条件
        z_c = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z_c)
        
        return x_hat, mu_q, sigma_q, mu_prior, sigma_prior
    
    def sample_prior(self, c_or_labels):
        """
        从条件先验分布采样：p(z|c) = N(μ_prior(c), σ_prior(c))
        Args:
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 如果提供了condition_encoder，则c_or_labels是标签，需要编码
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels
        
        mu_prior, sigma_prior = self.prior(c)
        z = self._reparameterize(mu_prior, sigma_prior)
        return z
    
    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


def loss_function_wo_prior(x_hat, x, mu_q, sigma_q):
    """
    损失函数 - 无Prior网络情况: p(z|c) = p(z) = N(0, I)
    
    ELBO = E_q(z|x,c)[log p(x|z,c)] - KL(q(z|x,c) || p(z))
    
    KL项: KL(q(z|x,c) || N(0, I))
    解析解: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    batch_size = x.shape[0]
    
    # 重构损失: -E_q(z|x,c)[log p(x|z,c)]
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度: KL(q(z|x,c) || N(0, I))
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(sigma_q + 1e-8) - mu_q ** 2 - sigma_q ** 2)
    
    return (recon_loss + kl_loss) / batch_size


def loss_function_prior(x_hat, x, mu_q, sigma_q, mu_prior, sigma_prior):
    """
    损失函数 - 有Prior网络情况: p(z|c) = N(μ_prior(c), σ_prior(c))
    
    ELBO = E_q(z|x,c)[log p(x|z,c)] - KL(q(z|x,c) || p(z|c))
    
    KL项: KL(q(z|x,c) || p(z|c))
    两个高斯分布之间的KL散度:
    KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 0.5
    """
    batch_size = x.shape[0]
    
    # 重构损失: -E_q(z|x,c)[log p(x|z,c)]
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度: KL(q(z|x,c) || p(z|c))
    # KL(N(μ_q,σ_q²) || N(μ_prior,σ_prior²))
    # = log(σ_prior/σ_q) + (σ_q² + (μ_q - μ_prior)²)/(2σ_prior²) - 0.5
    # 使用logvar形式: log(σ_prior/σ_q) = 0.5 * (logvar_prior - logvar_q)
    logvar_q = 2 * torch.log(sigma_q + 1e-8)
    logvar_prior = 2 * torch.log(sigma_prior + 1e-8)
    
    kl_loss = 0.5 * torch.sum(
        logvar_prior - logvar_q + 
        (sigma_q ** 2 + (mu_q - mu_prior) ** 2) / (2.0 * (sigma_prior ** 2 + 1e-8)) - 
        1.0
    )
    
    return (recon_loss + kl_loss) / batch_size


def loss_function(x_hat, x, mu, sigma, mu_prior=None, sigma_prior=None):
    """
    统一的损失函数接口
    如果mu_prior和sigma_prior为None，使用loss_function_wo_prior；否则使用loss_function_prior
    """
    if mu_prior is None or sigma_prior is None:
        return loss_function_wo_prior(x_hat, x, mu, sigma)
    else:
        return loss_function_prior(x_hat, x, mu, sigma, mu_prior, sigma_prior)

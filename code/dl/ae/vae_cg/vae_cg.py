"""
VAE with Classifier Guidance (CG) and Classifier-Free Guidance (CFG)
基于引导生成的VAE变种

实现两种引导方式：
1. Classifier Guidance (CG): 使用额外的分类器在潜在空间上引导生成
2. Classifier-Free Guidance (CFG): 训练条件和非条件模型，通过引导强度组合

参考CVAE的结构，但增加了引导机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OneHotEncoder(nn.Module):
    """One-hot编码器模块"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self, labels):
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
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
            self.condition_dim = embed_dim
        elif condition_encoding_type == 'onehot':
            self.encoder = OneHotEncoder(n_classes)
            self.condition_dim = n_classes
        else:
            raise ValueError(f"不支持的condition_encoding_type: {condition_encoding_type}，请选择 'onehot' 或 'embedding'")
    
    def forward(self, labels):
        """
        Args:
            labels: 标签张量，shape为 (batch_size,) 或 (batch_size, 1)
                   如果为None，返回None（用于无条件生成）
        
        Returns:
            编码后的条件张量，shape为 (batch_size, condition_dim) 或 None
        """
        if labels is None:
            return None
        
        # 确保labels是一维的
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        return self.encoder(labels)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim=None):
        """
        Args:
            condition_dim: 如果提供，则编码器接收条件信息
        """
        super().__init__()
        if condition_dim is not None:
            input_dim = input_dim + condition_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)
        self.condition_dim = condition_dim

    def forward(self, x, c=None):
        if self.condition_dim is not None:
            if c is not None:
                x = torch.cat((x, c), dim=1)
            else:
                # 当c为None时，使用零向量填充
                batch_size = x.shape[0]
                c_zero = torch.zeros(batch_size, self.condition_dim, device=x.device, dtype=x.dtype)
                x = torch.cat((x, c_zero), dim=1)
        h = F.relu(self.linear(x))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim=None):
        """
        Args:
            condition_dim: 如果提供，则解码器接收条件信息
        """
        super().__init__()
        if condition_dim is not None:
            latent_dim = latent_dim + condition_dim
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.condition_dim = condition_dim

    def forward(self, z, c=None):
        if self.condition_dim is not None:
            if c is not None:
                z = torch.cat((z, c), dim=1)
            else:
                # 当c为None时，使用零向量填充
                batch_size = z.shape[0]
                c_zero = torch.zeros(batch_size, self.condition_dim, device=z.device, dtype=z.dtype)
                z = torch.cat((z, c_zero), dim=1)
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat


class LatentClassifier(nn.Module):
    """
    潜在空间分类器：用于Classifier Guidance
    在潜在变量z上预测类别标签
    """
    def __init__(self, latent_dim, hidden_dim, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, z):
        h = F.relu(self.linear1(z))
        logits = self.linear2(h)
        return logits


class VAE_CG(nn.Module):
    """
    VAE with Classifier Guidance: 使用分类器引导生成
    
    训练阶段：
    - 训练VAE模型
    - 训练潜在空间分类器
    
    生成阶段：
    - 从先验采样z
    - 使用分类器梯度引导：z = z + guidance_scale * ∇_z log p(y|z)
    """
    def __init__(self, input_dim, output_dim, n_classes, hidden_dim, latent_dim,
                 condition_encoder=None, condition_dim=None):
        """
        Args:
            condition_encoder: ConditionEncoder实例，用于编码条件
            condition_dim: 条件维度（如果不提供condition_encoder，需要手动指定）
        """
        super().__init__()
        if condition_encoder is not None:
            condition_dim = condition_encoder.condition_dim
            self.condition_encoder = condition_encoder
        else:
            self.condition_encoder = None
            if condition_dim is None:
                raise ValueError("必须提供condition_encoder或condition_dim")
        
        self.n_classes = n_classes
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # VAE组件：使用条件
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, condition_dim)
        
        # 潜在空间分类器（用于CG）
        self.classifier = LatentClassifier(latent_dim, hidden_dim, n_classes)
    
    def forward(self, x, c_or_labels=None):
        """
        Args:
            x: 输入数据
            c_or_labels: 如果condition_encoder不为None，则为原始标签；否则为编码后的条件
        """
        # 编码条件
        if self.condition_encoder is not None:
            c = self.condition_encoder(c_or_labels) if c_or_labels is not None else None
        else:
            c = c_or_labels
        
        # 编码
        mu, sigma = self.encoder(x, c)
        z = self._reparameterize(mu, sigma)
        
        # 解码
        x_hat = self.decoder(z, c)
        
        # 分类器预测（用于训练）
        class_logits = self.classifier(z)
        
        return x_hat, mu, sigma, class_logits
    
    def sample_with_classifier_guidance(self, c_or_labels, guidance_scale=1.0, num_samples=1):
        """
        使用Classifier Guidance采样
        
        Args:
            c_or_labels: 目标类别标签 (batch_size,)
            guidance_scale: 引导强度
            num_samples: 采样数量（如果c_or_labels已提供，则忽略）
        
        Returns:
            z: 引导后的潜在变量
        """
        if isinstance(c_or_labels, torch.Tensor):
            num_samples = c_or_labels.shape[0]
            device = c_or_labels.device
        else:
            device = 'cpu'
            c_or_labels = torch.tensor([c_or_labels] * num_samples, device=device, dtype=torch.long)
        
        # 保存原始状态
        classifier_was_training = self.classifier.training
        # 确保分类器启用梯度计算（训练模式）
        self.classifier.train()
        
        # 确保分类器的参数可以计算梯度（虽然我们不会更新它们，但需要它们参与梯度计算）
        for param in self.classifier.parameters():
            param.requires_grad_(True)
        
        # 使用enable_grad确保梯度计算不被外部no_grad()影响
        with torch.enable_grad():
            try:
                # 从先验采样初始z
                z = torch.randn(num_samples, self.latent_dim, device=device, requires_grad=True)
                
                # 使用分类器梯度引导（Langevin动力学风格）
                for step in range(10):  # 多次迭代以增强引导效果
                    # 确保z需要梯度
                    if not z.requires_grad:
                        z = z.requires_grad_(True)
                    
                    # 前向传播
                    logits = self.classifier(z)
                    log_probs = F.log_softmax(logits, dim=1)
                    
                    # 计算目标类别的log概率（使用gather保持梯度）
                    indices = torch.arange(num_samples, device=device).unsqueeze(1)
                    labels_expanded = c_or_labels.unsqueeze(1)
                    target_log_probs = log_probs.gather(1, labels_expanded).squeeze(1)
                    target_log_prob = target_log_probs.sum()
                    
                    # 验证target_log_prob是否需要梯度
                    if not target_log_prob.requires_grad:
                        raise RuntimeError(
                            f"target_log_prob does not require grad. "
                            f"z.requires_grad={z.requires_grad}, "
                            f"logits.requires_grad={logits.requires_grad}"
                        )
                    
                    # 计算梯度
                    grad = torch.autograd.grad(
                        outputs=target_log_prob,
                        inputs=z,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True
                    )[0]
                    
                    # 更新z（使用no_grad避免构建梯度图）
                    with torch.no_grad():
                        # 更新z
                        z_new = z + guidance_scale * 0.01 * grad  # 小步长避免过度修正
                        # 添加少量噪声保持探索
                        if step < 9:  # 最后一步不加噪声
                            z_new = z_new + 0.01 * torch.randn_like(z_new)
                        
                        # 更新z并重新设置requires_grad
                        z = z_new.detach().requires_grad_(True)
                
                return z.detach()
            finally:
                # 恢复分类器的原始训练状态
                self.classifier.train(classifier_was_training)
    
    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


class VAE_CFG(nn.Module):
    """
    VAE with Classifier-Free Guidance: 训练条件和无条件模型，通过引导强度组合
    
    训练阶段：
    - 随机丢弃条件（unconditional_prob概率下使用None）
    - 训练模型同时支持条件和非条件生成
    
    生成阶段：
    - 同时计算条件和非条件的输出
    - 使用引导公式组合：ε_guided = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
    """
    def __init__(self, input_dim, output_dim, n_classes, hidden_dim, latent_dim,
                 condition_encoder=None, condition_dim=None, unconditional_prob=0.1):
        """
        Args:
            condition_encoder: ConditionEncoder实例
            condition_dim: 条件维度
            unconditional_prob: 训练时无条件生成的概率（用于CFG训练）
        """
        super().__init__()
        if condition_encoder is not None:
            condition_dim = condition_encoder.condition_dim
            self.condition_encoder = condition_encoder
        else:
            self.condition_encoder = None
            if condition_dim is None:
                raise ValueError("必须提供condition_encoder或condition_dim")
        
        self.n_classes = n_classes
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.unconditional_prob = unconditional_prob
        
        # VAE组件：支持条件和非条件
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, condition_dim)
    
    def forward(self, x, c_or_labels=None, force_unconditional=False):
        """
        Args:
            x: 输入数据
            c_or_labels: 条件标签，如果为None则无条件生成
            force_unconditional: 强制无条件生成（用于训练时的随机丢弃）
        """
        # 训练时随机丢弃条件（用于CFG）
        use_condition = not force_unconditional and c_or_labels is not None
        if self.training and not force_unconditional:
            # 随机丢弃条件
            if np.random.random() < self.unconditional_prob:
                use_condition = False
        
        # 编码条件
        if use_condition:
            if self.condition_encoder is not None:
                c = self.condition_encoder(c_or_labels)
            else:
                c = c_or_labels
        else:
            c = None
        
        # 编码
        mu, sigma = self.encoder(x, c)
        z = self._reparameterize(mu, sigma)
        
        # 解码
        x_hat = self.decoder(z, c)
        
        return x_hat, mu, sigma
    
    def sample_with_cfg(self, c_or_labels, guidance_scale=1.0, num_samples=1):
        """
        使用Classifier-Free Guidance采样
        
        Args:
            c_or_labels: 条件标签
            guidance_scale: 引导强度（w），通常 > 1.0
            num_samples: 采样数量
        
        Returns:
            x: 生成的图像
        """
        device = c_or_labels.device if c_or_labels is not None else 'cpu'
        
        # 从先验采样z
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # 编码条件
        if self.condition_encoder is not None and c_or_labels is not None:
            c = self.condition_encoder(c_or_labels)
        else:
            c = c_or_labels if c_or_labels is not None else None
        
        # 计算条件输出
        x_cond = self.decoder(z, c)
        
        # 计算无条件输出
        x_uncond = self.decoder(z, None)
        
        # CFG组合公式：x_guided = x_uncond + guidance_scale * (x_cond - x_uncond)
        # 或者更常见的形式：x_guided = (1 + guidance_scale) * x_cond - guidance_scale * x_uncond
        x_guided = x_uncond + guidance_scale * (x_cond - x_uncond)
        
        return x_guided, x_cond, x_uncond
    
    def _reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


def loss_function_cg(x_hat, x, mu, sigma, class_logits, labels, classification_weight=0.1):
    """
    VAE_CG的损失函数：包含重构损失、KL散度和分类损失
    
    Args:
        classification_weight: 分类损失的权重
    """
    batch_size = x.shape[0]
    
    # 重构损失
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(sigma + 1e-8) - mu ** 2 - sigma ** 2)
    
    # 分类损失（用于训练潜在空间分类器）
    classification_loss = F.cross_entropy(class_logits, labels, reduction='sum')
    
    total_loss = recon_loss + kl_loss + classification_weight * classification_loss
    
    return total_loss / batch_size


def loss_function_cfg(x_hat, x, mu, sigma):
    """
    VAE_CFG的损失函数：标准VAE损失（条件和无条件共享）
    """
    batch_size = x.shape[0]
    
    # 重构损失
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(sigma + 1e-8) - mu ** 2 - sigma ** 2)
    
    return (recon_loss + kl_loss) / batch_size


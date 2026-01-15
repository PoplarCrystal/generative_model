import torch
from torchvision import transforms
from tqdm import tqdm


class LatentDiffuser:
    """Latent Diffusion Model (LDM) with Classifier-Free Guidance (CFG)
    
    在潜在空间进行扩散，而不是在像素空间
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def diffusion(self, z_0, t):
        """前向扩散过程：向潜在向量添加噪声"""
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        # 对于潜在向量，不需要 view 成空间维度
        alpha_bar = alpha_bar.view(N, 1)  # (N, 1)

        noise = torch.randn_like(z_0, device=self.device)
        z_t = torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1 - alpha_bar) * noise
        return z_t, noise

    def denoise(self, model, z, t, labels=None, guidance_scale=3.0):
        """反向去噪过程：使用 Classifier-Free Guidance
        
        Args:
            model: UNet 模型（在潜在空间工作）
            z: 当前噪声潜在向量
            t: 时间步
            labels: 条件标签
            guidance_scale: CFG 引导强度（gamma），越大条件控制越强
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1)
        alpha_bar = alpha_bar.view(N, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1)

        model.eval()
        with torch.no_grad():
            # CFG: 同时计算条件预测和无条件预测
            eps_cond = model(z, t, labels)  # 条件预测
            eps_uncond = model(z, t, None)   # 无条件预测（labels=None）
            # CFG 公式: eps = eps_uncond + gamma * (eps_cond - eps_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        model.train()

        noise = torch.randn_like(z, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (z - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def sample(self, model, z_shape, labels=None, guidance_scale=3.0):
        """从随机噪声生成潜在向量（使用 Classifier-Free Guidance）
        
        Args:
            model: UNet 模型
            z_shape: 潜在向量的形状 (batch_size, latent_dim)
            labels: 可选的条件标签，如果为None则随机生成
            guidance_scale: CFG 引导强度（gamma），默认3.0
        """
        batch_size = z_shape[0]
        z = torch.randn(z_shape, device=self.device)
        
        # 如果未提供标签，随机生成
        if labels is None:
            labels = torch.randint(0, 10, (batch_size,), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1), desc='Sampling (LDM+CFG)'):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            z = self.denoise(model, z, t, labels, guidance_scale=guidance_scale)

        return z, labels



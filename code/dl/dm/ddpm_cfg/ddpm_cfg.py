import torch
from torchvision import transforms
from tqdm import tqdm


class Diffuser:
    """Classifier-Free Guidance (CFG) 扩散过程管理器"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def diffusion(self, x_0, t):
        """前向扩散过程：向图像添加噪声"""
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels=None, guidance_scale=3.0):
        """反向去噪过程：使用 Classifier-Free Guidance
        
        Args:
            model: UNet 模型
            x: 当前噪声图像
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
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            # CFG: 同时计算条件预测和无条件预测
            eps_cond = model(x, t, labels)  # 条件预测
            eps_uncond = model(x, t, None)   # 无条件预测（labels=None）
            # CFG 公式: eps = eps_uncond + gamma * (eps_cond - eps_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x):
        """将张量转换为 PIL 图像"""
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape, labels=None, guidance_scale=3.0):
        """从随机噪声生成图像（使用 Classifier-Free Guidance）
        
        Args:
            model: UNet 模型
            x_shape: 生成图像的形状 (batch_size, channels, height, width)
            labels: 可选的条件标签，如果为None则随机生成
            guidance_scale: CFG 引导强度（gamma），默认3.0
        """
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        
        # 如果未提供标签，随机生成
        if labels is None:
            labels = torch.randint(0, 10, (batch_size,), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1), desc='Sampling (CFG)'):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels, guidance_scale=guidance_scale)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels


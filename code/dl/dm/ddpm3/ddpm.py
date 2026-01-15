import torch
from torchvision import transforms


class Diffuser:
    """扩散过程管理器，支持三种预测目标：eps（噪声）、x0（原始数据）、score（得分）"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu', pred_objective='eps'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.pred_objective = pred_objective
        assert pred_objective in ['eps', 'x0', 'score'], (
            f"pred_objective must be 'eps', 'x0', or 'score', got {pred_objective}"
        )
        
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

    def _model_predictions(self, model, x, t, clip_x_start=False):
        """根据不同的预测目标，从模型输出得到预测的噪声和原始图像（私有辅助函数）"""
        model_output = model(x, t)
        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        if self.pred_objective == 'eps':
            # 预测噪声
            eps = model_output
            # 从噪声恢复 x0: x0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
            pred_x0 = (x - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
        elif self.pred_objective == 'x0':
            # 直接预测原始数据
            pred_x0 = model_output
            # 从 x0 计算噪声: eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1-alpha_bar)
            eps = (x - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha_bar
        elif self.pred_objective == 'score':
            # 预测得分（score = -eps / sqrt(1-alpha_bar)）
            score = model_output
            # 从 score 计算噪声: eps = -score * sqrt(1-alpha_bar)
            eps = -score * sqrt_one_minus_alpha_bar
            # 从噪声恢复 x0
            pred_x0 = (x - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
        
        if clip_x_start:
            pred_x0 = pred_x0.clamp(-1, 1)
            # 如果裁剪了 x0，重新计算 eps
            eps = (x - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha_bar
        
        return eps, pred_x0

    def denoise(self, model, x, t):
        """反向去噪过程：从噪声中恢复图像（支持三种预测目标）"""
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
            eps, pred_x0 = self._model_predictions(model, x, t, clip_x_start=True)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        # 使用预测的噪声计算后验均值（标准 DDPM 公式）
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

    def sample(self, model, x_shape):
        """从随机噪声生成图像
        
        Args:
            model: UNet 模型
            x_shape: 生成图像的形状 (batch_size, channels, height, width)
        """
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in range(self.num_timesteps, 0, -1):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images

    def sample_with_intermediate_steps(self, model, x_shape, step_interval, noise=None):
        """
        Generate a sample and return intermediate images at specified step intervals.
        This ensures all intermediate images are from the same denoising process.
        
        :param model: the model module.
        :param x_shape: the shape of the samples, (N, C, H, W).
        :param step_interval: interval between captured steps (e.g., num_timesteps/16).
        :param noise: if specified, the noise from the encoder to sample.
        :return: a tuple (final_images, intermediate_images_list, step_labels)
                 where intermediate_images_list contains images at step_interval, 2*step_interval, ...
        """
        batch_size = x_shape[0]
        if noise is not None:
            x = noise
        else:
            x = torch.randn(x_shape, device=self.device)
        
        # 计算要捕获的步数
        num_intermediates = int(self.num_timesteps / step_interval)
        steps_to_capture = [int(step_interval * i) for i in range(1, num_intermediates + 1)]
        
        intermediate_images = []
        step_labels = []
        current_step = 0

        for i in range(self.num_timesteps, 0, -1):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)
            current_step += 1
            
            # 如果当前步数在要捕获的列表中，保存图像
            if current_step in steps_to_capture:
                # 只保存第一张图像作为代表
                intermediate_img = self.reverse_to_img(x[0])
                intermediate_images.append(intermediate_img)
                step_labels.append(f'{current_step} steps')
        
        # 生成最终图像
        final_images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        
        return final_images, intermediate_images, step_labels


import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


class Diffuser:
    """Classifier Guidance (CG) 扩散过程管理器"""
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

    def denoise(self, model, x, t, labels=None, classifier=None, classifier_scale=1.0):
        """反向去噪过程：使用 Classifier Guidance
        
        Args:
            model: UNet 模型
            x: 当前噪声图像
            t: 时间步
            labels: 条件标签
            classifier: 分类器模型（用于CG）
            classifier_scale: CG 引导强度（s），越大条件控制越强
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

        # 模型评估模式：采样时需要使用 eval 模式
        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)  # 条件预测
        
        # Classifier Guidance: 使用分类器梯度引导
        if classifier is not None and labels is not None:
            # 需要确保 x 可以被求导
            x_grad = x.clone().detach().requires_grad_(True)
            classifier.eval()
            
            # 计算分类器预测
            logits = classifier(x_grad, t)
            log_probs = F.log_softmax(logits, dim=-1)
            # 选择目标类别的 log 概率
            selected = log_probs[range(len(logits)), labels.view(-1)]
            
            # 计算梯度（只对分类器进行反向传播）
            grad = torch.autograd.grad(
                selected.sum(), 
                x_grad, 
                create_graph=False, 
                retain_graph=False,
                only_inputs=True
            )[0]
            
            # 梯度裁剪：防止梯度过大导致采样不稳定
            # 计算每个样本的梯度范数
            grad_norm = torch.norm(grad.view(len(grad), -1), dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1) + 1e-8  # 避免除零
            
            # 梯度裁剪：限制梯度范数上限，防止过度引导
            # 当梯度范数 > 1.0 时，进行归一化；否则保持原样
            # 这样既能防止过大的梯度，又保留了正常梯度的信息
            max_grad_norm = 1.0
            grad_clipped = grad / torch.clamp(grad_norm, min=1.0)  # 只裁剪超过阈值的梯度
            
            # CG 公式: eps_guided = eps_cond + s * sigma_t * grad(log p(y|x_t))
            # 其中 sigma_t = sqrt(1 - alpha_bar)
            # 注意：即使加了梯度裁剪，也不建议使用过大的 classifier_scale (>2.0)
            sigma_t = torch.sqrt(1 - alpha_bar)
            
            # 应用引导：使用裁剪后的梯度
            eps = eps + classifier_scale * sigma_t * grad_clipped
            
            # 清理
            x_grad = None
        
        # 注意：采样时不改变模型状态，保持 eval 模式

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

    def sample(self, model, x_shape, labels=None, classifier=None, classifier_scale=1.0):
        """从随机噪声生成图像（使用 Classifier Guidance）
        
        Args:
            model: UNet 模型
            x_shape: 生成图像的形状 (batch_size, channels, height, width)
            labels: 可选的条件标签，如果为None则随机生成
            classifier: 分类器模型（用于CG）
            classifier_scale: CG 引导强度（s），默认1.0
        """
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        
        # 如果未提供标签，随机生成
        if labels is None:
            labels = torch.randint(0, 10, (batch_size,), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1), desc='Sampling (CG)'):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels, classifier=classifier, classifier_scale=classifier_scale)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels


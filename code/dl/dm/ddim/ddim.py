import torch
from torchvision import transforms


def compute_alpha(beta, t):
    """计算 alpha_bar，参考主目录 ddim 实现"""
    # 在 betas 前面加一个 0，所以索引需要 t + 1
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


class DDIMDiffuser:
    """DDIM 扩散过程管理器 - 支持加速采样（参考主目录 ddim 实现）"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu', eta=0.0):
        """
        Args:
            num_timesteps: 训练时的扩散步数（通常是1000）
            beta_start: beta 起始值
            beta_end: beta 结束值
            device: 设备
            eta: DDIM 采样参数，0.0 是完全确定性，1.0 是 DDPM
        """
        self.num_timesteps = num_timesteps
        self.device = device
        self.eta = eta
        
        # 计算 alpha 和 alpha_bar（与 DDPM 相同）
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def reverse_to_img(self, x):
        """将张量转换为 PIL 图像"""
        # 确保图像在 [0, 1] 范围内
        x = torch.clamp(x, 0.0, 1.0)
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def predict_x0_from_xt(self, model, x_t, t):
        """
        从 x_t 预测 x_0（参考主目录 ddim 实现）
        
        Args:
            model: UNet 模型
            x_t: 时间步 t 的图像
            t: 时间步 tensor（0-based，从0到num_timesteps-1）
        Returns:
            pred_x0: 预测的原始图像
            pred_eps: 预测的噪声
        """
        batch_size = x_t.size(0)
        
        # 使用 compute_alpha 计算 alpha_bar（0-based 索引）
        at = compute_alpha(self.betas, t.long())  # (N, 1, 1, 1)
        
        # 模型预测噪声
        # 注意：我们的 UNet 使用 1-based 时间步，所以需要转换
        # 主目录的模型使用 0-based，但我们的模型使用 1-based
        t_1based = t + 1  # 转换为 1-based
        
        model.eval()
        with torch.no_grad():
            pred_eps = model(x_t, t_1based.long())  # 我们的模型需要 long 类型
        model.train()
        
        # 从预测的噪声恢复 x_0（参考主目录实现）
        # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        pred_x0 = (x_t - pred_eps * (1 - at).sqrt()) / at.sqrt()
        
        return pred_x0, pred_eps

    def ddim_step(self, model, x_t, t, t_prev):
        """
        DDIM 单步采样：从 x_t 采样 x_{t_prev}（参考主目录 ddim 实现）
        
        Args:
            model: UNet 模型
            x_t: 时间步 t 的图像
            t: 当前时间步 tensor（0-based）
            t_prev: 前一个时间步 tensor（0-based，-1 表示结束，此时返回 pred_x0）
        Returns:
            x_prev: 采样得到的 x_{t_prev}
        """
        batch_size = x_t.size(0)
        
        # 预测 x_0
        pred_x0, pred_eps = self.predict_x0_from_xt(model, x_t, t)
        
        # 如果 t_prev = -1，表示采样结束，直接返回 pred_x0
        if (t_prev < 0).all():
            return pred_x0
        
        # 使用 compute_alpha 计算 alpha_bar（0-based 索引）
        at = compute_alpha(self.betas, t.long())  # (N, 1, 1, 1)
        at_next = compute_alpha(self.betas, t_prev.long())  # (N, 1, 1, 1)
        
        # 计算 c1 和 c2（参考主目录实现）
        # c1 = eta * sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
        c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        # c2 = sqrt((1 - at_next) - c1^2)
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        
        # DDIM 采样公式（参考主目录实现）
        # xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et
        # 当 eta=0 时，c1=0，使用零噪声实现确定性采样
        # 当 eta>0 时，使用随机噪声实现随机采样
        if self.eta > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)  # 确定性采样：使用零噪声
        x_prev = at_next.sqrt() * pred_x0 + c1 * noise + c2 * pred_eps
        
        return x_prev

    def sample(self, model, x_shape, ddim_steps=50):
        """
        DDIM 采样：从随机噪声生成图像（加速采样，参考主目录 ddim 实现）
        
        Args:
            model: UNet 模型
            x_shape: 生成图像的形状 (batch_size, channels, height, width)
            ddim_steps: DDIM 采样步数（通常远小于 num_timesteps）
        Returns:
            images: PIL 图像列表
        """
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        
        # 生成采样时间步序列（0-based，从 0 到 num_timesteps-1，均匀间隔）
        # 例如：如果 num_timesteps=1000, ddim_steps=50
        # 则采样时间步为 [0, 20, 40, ..., 980, 999]
        skip = self.num_timesteps // ddim_steps
        seq = list(range(0, self.num_timesteps, skip))  # 0-based: [0, 20, 40, ..., 980]
        seq_next = [-1] + list(seq[:-1])  # [-1, 0, 20, 40, ..., 960]
        
        # DDIM 采样循环（参考主目录实现）
        with torch.no_grad():
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.ones(batch_size, device=self.device) * i  # 0-based
                next_t = torch.ones(batch_size, device=self.device) * j  # 0-based，-1 表示结束
                x = self.ddim_step(model, x, t, next_t)
        
        # 确保最终图像在合理范围内
        # 注意：MNIST 图像应该在 [0, 1] 范围内，但采样过程中可能略有超出
        x = torch.clamp(x, 0.0, 1.0)
        
        # 转换为 PIL 图像
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images


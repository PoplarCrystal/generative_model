"""
Flow Matching: 基于 ODE 的生成模型。
参考 DDPM 的接口设计，实现从噪声分布到数据分布的流匹配。
- 概率路径：线性插值 x_t = (1-t)*x_0 + t*x_1，t in [0,1]
- 目标速度场：u_t = x_1 - x_0（常数）
- 训练：最小化 E[||v_theta(x_t, t) - (x_1 - x_0)||^2]
- 采样：从 x(0)~N(0,I) 解 ODE dx/dt = v_theta(x,t)，t: 0 -> 1
"""
import torch
from torchvision import transforms


class FlowMatcher:
    """Flow Matching 过程管理器：线性插值路径 + 速度场学习 + ODE 采样"""

    def __init__(self, device='cpu', time_embed_scale=1000, num_sample_steps=100):
        """
        Args:
            device: 设备
            time_embed_scale: 时间嵌入缩放，将 t in [0,1] 映射到 [1, time_embed_scale]，与 DDPM 的 num_timesteps 对齐以便复用 UNet
            num_sample_steps: 采样时 ODE 求解的步数
        """
        self.device = device
        self.time_embed_scale = time_embed_scale
        self.num_sample_steps = num_sample_steps

    def t_to_embed_index(self, t):
        """将连续时间 t in [0, 1] 转为 UNet 时间嵌入用的标量，范围 [1, time_embed_scale]"""
        # t: (N,) or scalar, in [0, 1]
        return 1.0 + t * (self.time_embed_scale - 1.0)

    def flow_forward(self, x_0, x_1, t):
        """
        前向流：在概率路径上采样 (x_t, target_velocity)。
        线性路径 x_t = (1-t)*x_0 + t*x_1，目标速度 u_t = x_1 - x_0。

        Args:
            x_0: 起点噪声 (N, C, H, W)，通常 N(0,I)
            x_1: 终点数据 (N, C, H, W)
            t: 时间 (N,) 或 (N,1,1,1)，范围 [0, 1]

        Returns:
            x_t: 当前时刻状态 (N, C, H, W)
            target_velocity: 目标速度 x_1 - x_0 (N, C, H, W)
        """
        t = t.to(self.device)
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)  # (N, 1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0
        return x_t, target_velocity

    def reverse_to_img(self, x):
        """将张量转换为 PIL 图像（与 DDPM 一致，数据假设在 [0,1] 或 [-1,1] 需与训练一致）"""
        # 若训练时数据 [0,1]，生成也在 [0,1]；若 [-1,1] 则先 clamp 再 (x+1)/2*255
        x = x.clamp(0, 1)
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape):
        """
        从随机噪声生成图像：解 ODE dx/dt = v_theta(x, t)，t 从 0 到 1。

        Args:
            model: 预测速度场的 UNet，输入 (x, t_embed)，t_embed 为 self.t_to_embed_index(t)
            x_shape: (batch_size, C, H, W)
        """
        batch_size = x_shape[0]
        dt = 1.0 / self.num_sample_steps
        x = torch.randn(x_shape, device=self.device)

        model.eval()
        with torch.no_grad():
            for step in range(self.num_sample_steps):
                t = torch.full((batch_size,), step * dt, device=self.device, dtype=torch.float32)
                t_embed = self.t_to_embed_index(t)
                t_embed_long = t_embed.long().clamp(1, self.time_embed_scale)
                v = model(x, t_embed_long)
                x = x + dt * v
        model.train()

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images

    def sample_with_intermediate_steps(self, model, x_shape, step_interval=None, noise=None):
        """
        生成样本并返回中间步的图像（与 DDPM 的 sample_with_intermediate_steps 对应）。

        Args:
            model: 速度场模型
            x_shape: (N, C, H, W)
            step_interval: 每隔多少步保存一次中间图；若为 None 则用 num_sample_steps // 16
            noise: 若给定则作为初始 x(0)，否则随机 N(0,I)

        Returns:
            final_images: 最终 PIL 列表
            intermediate_images: 中间步 PIL 列表
            step_labels: 每步标签
        """
        batch_size = x_shape[0]
        if step_interval is None:
            step_interval = max(1, self.num_sample_steps // 16)
        if noise is not None:
            x = noise.to(self.device)
        else:
            x = torch.randn(x_shape, device=self.device)

        num_intermediates = self.num_sample_steps // step_interval
        steps_to_capture = [step_interval * (i + 1) for i in range(num_intermediates)]
        intermediate_images = []
        step_labels = []
        current_step = 0
        dt = 1.0 / self.num_sample_steps

        model.eval()
        with torch.no_grad():
            for step in range(self.num_sample_steps):
                t = torch.full((batch_size,), step * dt, device=self.device, dtype=torch.float32)
                t_embed = self.t_to_embed_index(t)
                t_embed_long = t_embed.long().clamp(1, self.time_embed_scale)
                v = model(x, t_embed_long)
                x = x + dt * v
                current_step += 1
                if current_step in steps_to_capture:
                    intermediate_img = self.reverse_to_img(x[0])
                    intermediate_images.append(intermediate_img)
                    step_labels.append(f'{current_step} steps')
        model.train()

        final_images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return final_images, intermediate_images, step_labels

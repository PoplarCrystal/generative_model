import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D因果卷积层
    
    确保输出只依赖于当前时刻及之前的输入，不依赖未来信息
    使用膨胀卷积(Dilated Convolution)扩大感受野
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # Padding放在左边，模拟因果性
        # PyTorch的Conv1d padding是两边都加，我们需要自定义
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=self.padding, # 这里加了padding，输出变长
            dilation=dilation
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, seq_len)
        Returns:
            out: (batch_size, out_channels, seq_len)
        """
        out = self.conv(x)
        # 移除右侧多余的padding，只保留左侧的padding（历史信息），保持长度一致且具有因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """残差块：包含两个因果卷积层和跳跃连接"""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out + residual


class AR_CNN(nn.Module):
    """基于CNN（膨胀因果卷积）的自回归模型，用于生成MNIST图像
    
    使用WaveNet/PixelCNN风格的结构：
    1. 堆叠多层膨胀因果卷积，指数级扩大感受野
    2. 使用残差连接，加深网络
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=8, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影：将1维像素映射到hidden维度
        # 使用1x1卷积
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # 堆叠残差块，dilation指数增长 (1, 2, 4, 8, ...)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(
                ResidualBlock(hidden_dim, kernel_size, dilation)
            )
            
        # 输出投影：将hidden维度映射回1维像素
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, seq_len, input_dim) Logits
        """
        # 调整维度以适应Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # 1. 输入投影
        x = self.input_proj(x)
        
        # 2. 通过残差块
        for layer in self.layers:
            x = layer(x)
            
        # 3. 输出投影
        x = self.output_proj(x)
        
        # 调整回序列维度: (B, C, L) -> (B, L, C)
        return x.transpose(1, 2)
    
    def generate(self, batch_size=1, seq_len=784, device='cpu', temperature=1.0):
        """自回归生成图像
        
        Args:
            batch_size: 生成图像的数量
            seq_len: 序列长度（MNIST: 28*28=784）
            device: 设备
            temperature: 采样温度
        Returns:
            generated_seq: (batch_size, seq_len, input_dim)
        """
        self.eval()
        with torch.no_grad():
            # 1. 初始化画布 (全0)
            generated_seq = torch.zeros(batch_size, seq_len, self.input_dim, device=device)
            
            # 2. 逐个像素生成
            # CNN必须每次重新计算整个序列，因为没有Hidden State
            for t in range(1, seq_len):
                # A: 获取当前已生成的序列（到位置t-1）
                current_seq = generated_seq[:, :t, :]
                
                # B: 前向传播
                # output shape: (batch, t, input_dim)
                output = self.forward(current_seq)
                
                # C: 取最后一个位置的预测 (Logits)
                logits = output[:, -1, :]
                
                # D: 采样
                probs = torch.sigmoid(logits / temperature)
                next_pixel = torch.bernoulli(probs)
                
                # E: 填入画布
                generated_seq[:, t, :] = next_pixel
            
            return generated_seq


def loss_function(pred, target):
    """计算BCEWithLogits损失
    
    Args:
        pred: (batch_size, seq_len, input_dim) 模型输出的Logits
        target: (batch_size, seq_len, input_dim) 目标像素值[0,1]
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat)
    return loss
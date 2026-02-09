import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D因果卷积层
    
    确保输出只依赖于当前时刻及之前的输入，不依赖未来信息
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=self.padding,
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
        # 移除右侧的padding，保持因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class AR_CNN(nn.Module):
    """基于CNN（因果卷积）的自回归模型，用于生成MNIST图像
    
    使用因果卷积建模序列依赖关系，每个像素值被量化为256个级别（0-255）
    """
    def __init__(self, vocab_size=256, embedding_dim=128, hidden_dim=256, num_layers=4, kernel_size=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # 嵌入层：将像素值（0-255）映射到embedding空间
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 因果卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一层：从embedding到hidden
        self.conv_layers.append(
            CausalConv1d(embedding_dim, hidden_dim, kernel_size)
        )
        
        # 中间层：hidden到hidden
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size)
            )
        
        # 最后一层：从hidden到vocab_size
        self.conv_layers.append(
            CausalConv1d(hidden_dim, vocab_size, kernel_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) 输入序列，每个元素是像素值（0-255）
        Returns:
            logits: (batch_size, seq_len, vocab_size) 每个位置的下一个像素值的logits
        """
        # 嵌入
        x_emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # 转换为卷积格式：(batch_size, channels, seq_len)
        x_conv = x_emb.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # 通过因果卷积层
        for i, conv_layer in enumerate(self.conv_layers):
            x_conv = conv_layer(x_conv)
            # 中间层使用ReLU激活
            if i < len(self.conv_layers) - 1:
                x_conv = F.relu(x_conv)
        
        # 转换回序列格式：(batch_size, seq_len, vocab_size)
        logits = x_conv.transpose(1, 2)
        
        return logits
    
    def generate(self, batch_size=1, seq_len=784, device='cpu', temperature=1.0):
        """自回归生成图像
        
        Args:
            batch_size: 生成图像的数量
            seq_len: 序列长度（MNIST: 28*28=784）
            device: 设备
            temperature: 采样温度，控制随机性
        Returns:
            generated: (batch_size, seq_len) 生成的像素序列
        """
        self.eval()
        with torch.no_grad():
            # 初始化：全零序列，让模型生成所有像素
            generated = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            
            # 自回归生成
            for t in range(seq_len):
                # 获取当前序列（到位置t）
                x_in = generated[:, :t+1]  # (batch_size, t+1)
                
                # 获取当前序列的logits
                logits = self.forward(x_in)  # (batch_size, t+1, vocab_size)
                
                # 取最后一个位置的logits
                next_logits = logits[:, -1, :]  # (batch_size, vocab_size)
                next_logits = next_logits / temperature
                
                # 采样
                probs = F.softmax(next_logits, dim=-1)
                next_pixel = torch.multinomial(probs, 1).squeeze(-1)  # (batch_size,)
                
                # 更新序列
                generated[:, t] = next_pixel
            
            return generated


def loss_function(logits, targets):
    """计算交叉熵损失
    
    Args:
        logits: (batch_size, seq_len, vocab_size) 模型输出的logits
        targets: (batch_size, seq_len) 目标像素值
    Returns:
        loss: 平均交叉熵损失
    """
    # 将logits和targets展平
    logits_flat = logits.reshape(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch_size * seq_len)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    return loss

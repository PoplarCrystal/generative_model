import torch
import torch.nn as nn
import torch.nn.functional as F


class AR_RNN(nn.Module):
    """基于RNN的自回归模型，用于生成MNIST图像
    
    将28x28的图像展平为784个像素的序列，使用RNN建模序列依赖关系
    每个像素值被量化为256个级别（0-255）
    """
    def __init__(self, vocab_size=256, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层：将像素值（0-255）映射到embedding空间
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层：建模序列依赖
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层：预测下一个像素值的分布
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) 输入序列，每个元素是像素值（0-255）
        Returns:
            logits: (batch_size, seq_len, vocab_size) 每个位置的下一个像素值的logits
        """
        # 嵌入
        x_emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x_emb)  # (batch_size, seq_len, hidden_dim)
        
        # 输出logits
        logits = self.output(lstm_out)  # (batch_size, seq_len, vocab_size)
        
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
            h = None  # LSTM隐藏状态
            
            # 自回归生成
            for t in range(seq_len):
                # 获取当前序列（到位置t）
                x_in = generated[:, :t+1]  # (batch_size, t+1)
                
                # 嵌入
                x_emb = self.embedding(x_in)  # (batch_size, t+1, embedding_dim)
                
                # LSTM前向传播
                if t == 0:
                    # 第一次：用完整序列初始化LSTM状态
                    lstm_out, h = self.lstm(x_emb)  # h = (h_n, c_n)
                else:
                    # 之后：只输入最新token，利用隐藏状态（更高效）
                    lstm_out, h = self.lstm(x_emb[:, -1:], h)  # (batch_size, 1, hidden_dim)
                
                # 预测下一个像素
                logits = self.output(lstm_out[:, -1, :])  # (batch_size, vocab_size)
                logits = logits / temperature
                
                # 采样
                probs = F.softmax(logits, dim=-1)
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

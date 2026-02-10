import torch
import torch.nn as nn
import torch.nn.functional as F


class AR_RNN(nn.Module):
    """基于RNN的自回归模型，用于生成MNIST图像
    
    将28x28的图像展平为784个像素的序列，使用RNN建模序列依赖关系
    每个像素值是连续的[0, 1]范围
    """
    def __init__(self, input_dim=1, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入层：将像素值映射到hidden维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM层：建模序列依赖
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层：预测下一个像素值的Logits (未经过Sigmoid)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 输入序列
        Returns:
            output: (batch_size, seq_len, input_dim) 每个位置的下一个像素值的Logits
        """
        # 投影到hidden维度
        x_proj = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x_proj)  # (batch_size, seq_len, hidden_dim)
        
        # 输出下一个像素值
        output = self.output(lstm_out)  # (batch_size, seq_len, input_dim)
        
        return output
    
    def generate(self, batch_size=1, seq_len=784, device='cpu', temperature=1.0):
        """自回归生成图像
        
        Args:
            batch_size: 生成图像的数量
            seq_len: 序列长度（MNIST: 28*28=784）
            device: 设备
            temperature: 采样温度，控制多样性
        Returns:
            generated_seq: (batch_size, seq_len, input_dim) 生成的像素序列
        """
        self.eval()
        with torch.no_grad():
            # 1. 初始化画布 (全0)
            generated_seq = torch.zeros(batch_size, seq_len, self.input_dim, device=device)
            
            # 2. 记忆状态 (Hidden State)，初始为空
            hidden_state = None 

            # 3. 逐个像素生成
            # t 从 1 开始，因为第 0 个像素我们默认为 0
            for t in range(1, seq_len):
                # A: 准备当前步的输入 (上一个时刻生成的像素)
                last_pixel = generated_seq[:, t-1:t, :]  # Shape: (batch, 1, 1)

                # B: 特征投影 (将像素变成特征)
                current_feature = self.input_proj(last_pixel) # Shape: (batch, 1, hidden_dim)

                # C: LSTM 更新 (只处理当前token，利用hidden_state传递历史)
                lstm_out, hidden_state = self.lstm(current_feature, hidden_state)

                # D: 预测下一个像素 (Logits)
                logits = self.output(lstm_out[:, -1, :])
                
                # E: 采样
                probs = torch.sigmoid(logits / temperature)
                next_pixel = torch.bernoulli(probs)
                
                # F: 填入画布
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

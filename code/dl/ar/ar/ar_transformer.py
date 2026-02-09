import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码，为序列添加位置信息"""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attn_mask: 注意力掩码，用于因果注意力（防止看到未来信息）
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # 前馈网络
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class AR_Transformer(nn.Module):
    """基于Transformer的自回归模型，用于生成MNIST图像
    
    使用Transformer的自注意力机制建模序列依赖关系，每个像素值被量化为256个级别（0-255）
    """
    def __init__(self, vocab_size=256, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1, max_len=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 嵌入层：将像素值（0-255）映射到d_model空间
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层：预测下一个像素值的分布
        self.output = nn.Linear(d_model, vocab_size)
        
    def _generate_causal_mask(self, seq_len, device):
        """生成因果注意力掩码，防止看到未来信息"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x, use_causal_mask=True):
        """
        Args:
            x: (batch_size, seq_len) 输入序列，每个元素是像素值（0-255）
            use_causal_mask: 是否使用因果掩码（训练时通常为True）
        Returns:
            logits: (batch_size, seq_len, vocab_size) 每个位置的下一个像素值的logits
        """
        # 嵌入
        x_emb = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x_emb = self.pos_encoder(x_emb)
        
        # 生成因果掩码
        attn_mask = None
        if use_causal_mask:
            attn_mask = self._generate_causal_mask(x.size(1), x.device)
        
        # 通过Transformer层
        out = x_emb
        for transformer_block in self.transformer_blocks:
            out = transformer_block(out, attn_mask=attn_mask)
        
        # 输出logits
        logits = self.output(out)  # (batch_size, seq_len, vocab_size)
        
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
                
                # 获取当前序列的logits（使用因果掩码）
                logits = self.forward(x_in, use_causal_mask=True)  # (batch_size, t+1, vocab_size)
                
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

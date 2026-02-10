当然可以！以下提供 自回归（AR） 和 条件自回归（CAR） 模型的最小可运行 PyTorch 代码示例，涵盖三种主流架构：

- RNN（LSTM）
- CNN（因果卷积）
- Transformer（带因果掩码）

所有代码均包含：
- 模型定义
- 训练步骤（使用 teacher forcing）
- 推理生成（自回归采样）
- 支持无条件（AR）和有条件（CAR）两种模式

💡 假设输入为离散 token 序列（如 MNIST 像素展平后量化为 0–255，或文本词表），输出为类别分布。

公共设置（所有模型共享）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

超参数（可根据任务调整）
vocab_size = 256      # 例如：8-bit 像素值 (0-255)
seq_len = 784         # MNIST: 28x28 = 784
batch_size = 32
d_model = 128
num_epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

模拟数据：x 是 [B, T] 的 long 张量（token IDs），c 是 [B, C] 的条件向量（可选）
def get_dummy_data(batch_size, seq_len, vocab_size, cond_dim=None):
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).long()
    c = None
    if cond_dim is not None:
        c = torch.randn(batch_size, cond_dim)
    return x.to(device), c.to(device) if c is not None else None

1. RNN-based AR / CAR

class RNN_AR(nn.Module):
    def init(self, vocab_size, d_model=128, n_layers=2, cond_dim=None):
        super().init()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim else None
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)
        self.cond_dim = cond_dim

    def forward(self, x, c=None):
        # x: [B, T], c: [B, C] or None
        B, T = x.shape
        x_emb = self.embedding(x)  # [B, T, D]

        if c is not None:
            assert self.cond_proj is not None
            c_emb = self.cond_proj(c).unsqueeze(1)  # [B, 1, D]
            x_emb = x_emb + c_emb  # 广播到所有时间步

        h, _ = self.lstm(x_emb)  # [B, T, D]
        logits = self.head(h)   # [B, T, V]
        return logits

    def generate(self, c=None, max_len=784, temperature=1.0):
        # 自回归生成
        B = c.shape[0] if c is not None else 1
        device = next(self.parameters()).device
        x_gen = torch.zeros(B, max_len).long().to(device)
        h = None  # LSTM 隐藏状态

        for t in range(max_len):
            x_in = x_gen[:, :t+1]  # [B, t+1]
            x_emb = self.embedding(x_in)

            if c is not None and t == 0:
                c_emb = self.cond_proj(c).unsqueeze(1)  # [B, 1, D]
                x_emb = x_emb + c_emb

            if t == 0:
                out, h = self.lstm(x_emb)
            else:
                out, h = self.lstm(x_emb[:, -1:], h)  # 只输入最新 token

            logits = self.head(out[:, -1]) / temperature  # [B, V]
            probs = F.softmax(logits, dim=-1)
            x_gen[:, t] = torch.multinomial(probs, 1).squeeze(-1)

        return x_gen

2. CNN-based AR / CAR（因果卷积）

class CausalConv1d(nn.Module):
    def init(self, in_channels, out_channels, kernel_size):
        super().init()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # x: [B, D, T]
        x = F.pad(x, (self.pad, 0))  # 左填充
        return self.conv(x)  # [B, D, T]

class CNN_AR(nn.Module):
    def init(self, vocab_size, d_model=128, n_layers=5, kernel_size=3, cond_dim=None):
        super().init()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim else None
        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model, kernel_size) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        self.cond_dim = cond_dim

    def forward(self, x, c=None):
        # x: [B, T], c: [B, C] or None
        x = self.embedding(x).transpose(1, 2)  # [B, D, T]

        if c is not None:
            c_emb = self.cond_proj(c).unsqueeze(-1)  # [B, D, 1]
            x = x + c_emb  # 广播

        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)  # [B, T, D]
        logits = self.head(x)  # [B, T, V]
        return logits

    def generate(self, c=None, max_len=784, temperature=1.0):
        B = c.shape[0] if c is not None else 1
        device = next(self.parameters()).device
        x_gen = torch.zeros(B, max_len).long().to(device)

        for t in range(max_len):
            x_in = x_gen[:, :t+1]  # [B, t+1]
            x_emb = self.embedding(x_in).transpose(1, 2)  # [B, D, t+1]

            if c is not None:
                c_emb = self.cond_proj(c).unsqueeze(-1)  # [B, D, 1]
                x_emb = x_emb + c_emb

            x_out = x_emb
            for conv in self.convs:
                x_out = F.relu(conv(x_out))
            logits = self.head(x_out.transpose(1, 2)[:, -1]) / temperature  # [B, V]
            probs = F.softmax(logits, dim=-1)
            x_gen[:, t] = torch.multinomial(probs, 1).squeeze(-1)

        return x_gen

3. Transformer-based AR / CAR（因果注意力）

class CausalTransformer(nn.Module):
    def init(self, vocab_size, d_model=128, nhead=8, num_layers=3, cond_dim=None):
        super().init()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # 支持长序列
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.cond_dim = cond_dim

    def forward(self, x, c=None):
        # x: [B, T], c: [B, C] or None
        B, T = x.shape
        x = self.embedding(x) + self.pos_encoding[:, :T]  # [B, T, D]

        if c is not None:
            c_emb = self.cond_proj(c).unsqueeze(1)  # [B, 1, D]
            x = torch.cat([c_emb, x], dim=1)  # [B, T+1, D]
            src_len = T + 1
        else:
            src_len = T

        # 创建因果掩码（上三角为 -inf）
        mask = torch.triu(torch.ones(src_len, src_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)

        out = self.transformer(x, mask=mask)  # [B, T(+1), D]

        if c is not None:
            out = out[:, 1:]  # 去掉条件 token

        logits = self.head(out)  # [B, T, V]
        return logits

    def generate(self, c=None, max_len=784, temperature=1.0):
        B = c.shape[0] if c is not None else 1
        device = next(self.parameters()).device
        x_gen = torch.zeros(B, max_len).long().to(device)

        # 缓存 K/V 以加速（简化版：不实现 KV cache，每次重算）
        for t in range(max_len):
            x_in = x_gen[:, :t+1]  # [B, t+1]
            logits = self.forward(x_in, c)  # [B, t+1, V]
            next_logits = logits[:, -1] / temperature  # [B, V]
            probs = F.softmax(next_logits, dim=-1)
            x_gen[:, t] = torch.multinomial(probs, 1).squeeze(-1)

        return x_gen

训练与推理示例（以 Transformer 为例）

初始化模型（无条件 AR）
model = CausalTransformer(vocab_size=vocab_size, d_model=d_model).to(device)

或有条件 CAR（假设条件维度为 64）
model = CausalTransformer(vocab_size=vocab_size, d_model=d_model, cond_dim=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

训练循环
for epoch in range(num_epochs):
    model.train()
    x, c = get_dummy_data(batch_size, seq_len, vocab_size, cond_dim=64 if model.cond_dim else None)
    
    # Teacher forcing: 输入 x[:, :-1], 目标 x[:, 1:]
    input_seq = x[:, :-1]
    target_seq = x[:, 1:]
    
    logits = model(input_seq, c)  # [B, T-1, V]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

推理生成
model.eval()
with torch.no_grad():
    if model.cond_dim:
        c_test = torch.randn(1, model.cond_dim).to(device)
        gen = model.generate(c=c_test, max_len=seq_len)
    else:
        gen = model.generate(max_len=seq_len)
    print("Generated sequence shape:", gen.shape)  # [1, 784]

说明与扩展建议

1. 离散 vs 连续输出：
   - 上述代码假设离散输出（分类分布）；
   - 若用于连续数据（如音频波形），需将 head 改为输出均值/方差，并使用 Gaussian NLL Loss。

2. 效率优化：
   - Transformer 推理时可添加 KV Cache 避免重复计算；
   - CNN 可用 空洞因果卷积（WaveNet）扩大感受野。

3. 条件融合方式：
   - 当前使用简单加法或拼接；
   - 更高级方式：cross-attention（Transformer）、FiLM（CNN/RNN）。

4. 实际应用：
   - PixelCNN：2D 因果卷积（非 1D）；
   - GPT：标准因果 Transformer；
   - Music Transformer：相对位置编码 + 条件控制。

这些代码提供了清晰、可运行的起点。您可以根据具体任务（MNIST、文本、音频）调整 vocab_size、seq_len、条件维度等。如需完整 Colab 笔记本、MNIST 训练示例，或支持 2D 图像的 PixelCNN 实现，请随时告诉我！
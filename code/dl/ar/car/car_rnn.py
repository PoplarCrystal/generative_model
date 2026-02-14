"""条件自回归 (Conditional AR) 模型 - RNN 实现

基于 AR_RNN 架构，参考 cddpm 的条件注入方式，支持标签条件生成。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAR_RNN(nn.Module):
    """条件自回归 RNN 模型，用于条件生成 MNIST 图像

    在 AR_RNN (LSTM) 基础上增加标签条件：
    1. 使用 nn.Embedding 将类别标签嵌入为向量
    2. 将标签嵌入加到每步的输入特征上，实现条件控制
    3. 支持按指定类别生成数字图像
    """
    def __init__(
        self,
        input_dim=1,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        num_labels=10,
        label_embed_dim=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_labels = num_labels

        if label_embed_dim is None:
            label_embed_dim = hidden_dim

        # 标签嵌入（参考 cddpm）
        self.label_emb = nn.Embedding(num_labels, label_embed_dim)
        if label_embed_dim != hidden_dim:
            self.label_proj = nn.Linear(label_embed_dim, hidden_dim)
        else:
            self.label_proj = nn.Identity()

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, labels):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            labels: (batch_size,) 类别标签
        Returns:
            output: (batch_size, seq_len, input_dim) Logits
        """
        x_proj = self.input_proj(x)  # (B, L, hidden_dim)

        # 注入标签条件：与 CAR_CNN/cddpm 一致，加到特征上
        label_emb = self.label_emb(labels)           # (B, label_embed_dim)
        label_emb = self.label_proj(label_emb)       # (B, hidden_dim)
        label_emb = label_emb.unsqueeze(1)          # (B, 1, hidden_dim)
        x_proj = x_proj + label_emb                 # (B, L, hidden_dim)

        lstm_out, _ = self.lstm(x_proj)            # (B, L, hidden_dim)
        return self.output(lstm_out)                # (B, L, input_dim)

    def generate(
        self,
        batch_size=1,
        seq_len=784,
        labels=None,
        device='cpu',
        temperature=1.0,
    ):
        """条件自回归生成图像

        Args:
            batch_size: 生成图像的数量
            seq_len: 序列长度（MNIST: 28*28=784）
            labels: (batch_size,) 条件标签，若为 None 则随机采样
            device: 设备
            temperature: 采样温度
        Returns:
            generated_seq: (batch_size, seq_len, input_dim)
            labels: 实际使用的标签
        """
        self.eval()
        with torch.no_grad():
            if labels is None:
                labels = torch.randint(
                    0, self.num_labels, (batch_size,), device=device
                )

            # 每步的标签条件（固定不变）
            label_emb = self.label_emb(labels)
            label_emb = self.label_proj(label_emb)  # (B, hidden_dim)

            generated_seq = torch.zeros(
                batch_size, seq_len, self.input_dim, device=device
            )
            hidden_state = None

            for t in range(1, seq_len):
                last_pixel = generated_seq[:, t - 1 : t, :]  # (B, 1, input_dim)
                current_feature = self.input_proj(last_pixel)  # (B, 1, hidden_dim)
                current_feature = current_feature + label_emb.unsqueeze(1)

                lstm_out, hidden_state = self.lstm(current_feature, hidden_state)
                logits = self.output(lstm_out[:, -1, :])
                probs = torch.sigmoid(logits / temperature)
                next_pixel = torch.bernoulli(probs)
                generated_seq[:, t, :] = next_pixel

            return generated_seq, labels


def loss_function(pred, target):
    """计算 BCEWithLogits 损失

    Args:
        pred: (batch_size, seq_len, input_dim) 模型输出的 Logits
        target: (batch_size, seq_len, input_dim) 目标像素值 [0, 1]
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    return F.binary_cross_entropy_with_logits(pred_flat, target_flat)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGenerator(nn.Module):
    """条件生成器：从噪声和标签生成图像"""
    def __init__(self, noise_dim, num_labels, label_embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_labels = num_labels
        self.label_embed = nn.Embedding(num_labels, label_embed_dim)
        # 输入：噪声 + 标签嵌入
        self.linear1 = nn.Linear(noise_dim + label_embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, labels):
        # 标签嵌入
        label_emb = self.label_embed(labels)  # [B, label_embed_dim]
        # 拼接噪声和标签嵌入
        z_cond = torch.cat([z, label_emb], dim=1)  # [B, noise_dim + label_embed_dim]
        h = F.relu(self.linear1(z_cond))
        h = F.relu(self.linear2(h))
        x = torch.sigmoid(self.linear3(h))
        return x


class ConditionalDiscriminator(nn.Module):
    """条件判别器：判断输入图像和标签的组合是真实还是生成"""
    def __init__(self, input_dim, num_labels, label_embed_dim, hidden_dim):
        super().__init__()
        self.num_labels = num_labels
        self.label_embed = nn.Embedding(num_labels, label_embed_dim)
        # 输入：图像 + 标签嵌入
        self.linear1 = nn.Linear(input_dim + label_embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, labels):
        # 标签嵌入
        label_emb = self.label_embed(labels)  # [B, label_embed_dim]
        # 拼接图像和标签嵌入
        x_cond = torch.cat([x, label_emb], dim=1)  # [B, input_dim + label_embed_dim]
        h = F.relu(self.linear1(x_cond))
        h = F.relu(self.linear2(h))
        out = torch.sigmoid(self.linear3(h))
        return out


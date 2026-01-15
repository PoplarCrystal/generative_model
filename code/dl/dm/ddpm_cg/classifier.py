import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    分类器：用于 Classifier Guidance
    输入：噪声图像 x_t 和时间步 t
    输出：类别 logits
    """
    def __init__(self, img_size=28, in_channels=1, num_classes=10, time_embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.time_embed_dim = time_embed_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 图像特征提取
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 融合时间和图像特征
        self.fc = nn.Sequential(
            nn.Linear(128 + time_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, t):
        """
        Args:
            x: [B, C, H, W] 噪声图像
            t: [B] 时间步
        Returns:
            logits: [B, num_classes] 类别 logits
        """
        B = x.shape[0]
        
        # 时间嵌入
        t = t.float().view(B, 1) / 1000.0  # 归一化到 [0, 1]
        t_emb = self.time_mlp(t)  # [B, time_embed_dim]
        
        # 图像特征
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # [B, 128, 1, 1]
        x = x.view(B, -1)  # [B, 128]
        
        # 融合特征
        combined = torch.cat([x, t_emb], dim=1)  # [B, 128 + time_embed_dim]
        logits = self.fc(combined)  # [B, num_classes]
        
        return logits


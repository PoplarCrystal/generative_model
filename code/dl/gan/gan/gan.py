import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """生成器：从噪声生成图像"""
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(noise_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.linear4 = nn.Linear(hidden_dim // 2, output_dim)
        
        # 权重初始化：使用Xavier初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        h = F.relu(self.bn1(self.linear1(z)))
        h = F.relu(self.bn2(self.linear2(h)))
        h = F.relu(self.bn3(self.linear3(h)))
        x = torch.sigmoid(self.linear4(h))
        # 确保输出在[0,1]范围内
        x = torch.clamp(x, 0.0, 1.0)
        return x


class Discriminator(nn.Module):
    """判别器：判断输入是真实图像还是生成图像"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(hidden_dim // 4, 1)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        h = self.dropout1(F.leaky_relu(self.linear1(x), 0.2))
        h = self.dropout2(F.leaky_relu(self.linear2(h), 0.2))
        h = self.dropout3(F.leaky_relu(self.linear3(h), 0.2))
        out = torch.sigmoid(self.linear4(h))
        return out


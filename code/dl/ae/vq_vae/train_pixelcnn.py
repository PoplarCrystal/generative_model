import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml

# 导入模块 - 支持从不同路径运行
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
# 脚本文件总是在 vq_vae 目录下，所以直接添加当前目录到路径
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from vq_vae import VQVAE
from pixelcnn_model import PixelCNNWithEmbedding

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取 VQ-VAE 参数
input_dim = config['model']['input_dim']
dim = config['model']['dim']
n_embedding = config['model']['n_embedding']

# 从配置文件读取 PixelCNN 参数
pixelcnn_n_blocks = config['pixelcnn']['n_blocks']
pixelcnn_dim = config['pixelcnn']['dim']
pixelcnn_linear_dim = config['pixelcnn']['linear_dim']
pixelcnn_use_bn = config['pixelcnn']['use_bn']

# 从配置文件读取训练参数
epochs = config['pixelcnn_training']['epochs']
learning_rate = config['pixelcnn_training']['learning_rate']
batch_size = config['pixelcnn_training']['batch_size']

# 从配置文件读取模型文件路径
vqvae_model_name = config['model_file']['model_name']
pixelcnn_model_name = config['pixelcnn_model_file']['model_name']
# 模型保存路径：与代码文件路径一致
vqvae_model_path = os.path.join(current_dir, vqvae_model_name)
pixelcnn_model_path = os.path.join(current_dir, pixelcnn_model_name)

# 获取data文件夹的绝对路径
data_dir_rel = config['data']['data_dir']
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 第二步：加载 VQ-VAE 模型
vqvae = VQVAE(input_dim, dim, n_embedding).to(device)
try:
    vqvae.load_state_dict(torch.load(vqvae_model_path, weights_only=True))
    print(f'VQ-VAE 模型已加载: {vqvae_model_path}')
    vqvae.eval()
except FileNotFoundError:
    print(f'错误: 未找到 VQ-VAE 模型文件 {vqvae_model_path}')
    print('请先训练 VQ-VAE 模型')
    exit(1)

# 第三步：构建 PixelCNN 模型
pixelcnn = PixelCNNWithEmbedding(
    n_blocks=pixelcnn_n_blocks,
    p=pixelcnn_dim,
    linear_dim=pixelcnn_linear_dim,
    bn=pixelcnn_use_bn,
    color_level=n_embedding  # 输出类别数等于 codebook 大小
).to(device)

optimizer = optim.Adam(pixelcnn.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# 第四步：数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 第五步：训练 PixelCNN
print('\n开始训练 PixelCNN...')
train_losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for x, labels in pbar:
        x = x.to(device)  # (N, 1, 28, 28
        
        # 使用 VQ-VAE 编码得到离散索引
        with torch.no_grad():
            discrete_indices = vqvae.encode(x)  # (N, H, W) = (N, 7, 7)
        
        # PixelCNN 预测下一个索引
        predict_indices = pixelcnn(discrete_indices)  # (N, n_embedding, H, W)
        
        # 计算损失（交叉熵）
        # 需要将 (N, C, H, W) 转换为 (N*H*W, C) 和 (N*H*W,)
        N, C, H, W = predict_indices.shape
        predict_indices_flat = predict_indices.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        target_indices_flat = discrete_indices.reshape(-1)  # (N*H*W,)
        
        loss = loss_fn(predict_indices_flat, target_indices_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        cnt += 1
        pbar.set_postfix({'loss': f'{loss_sum/cnt:.4f}'})
    
    train_loss_avg = loss_sum / cnt
    train_losses.append(train_loss_avg)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}')

# 保存 PixelCNN 模型
torch.save(pixelcnn.state_dict(), pixelcnn_model_path)
print(f'\nPixelCNN 模型已保存: {pixelcnn_model_path}')

# 第六步：可视化训练损失
if len(train_losses) > 0:
    epochs_list = list(range(1, len(train_losses) + 1))
    plt.plot(epochs_list, train_losses, marker='o', linestyle='-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('PixelCNN Training Loss')
    plt.show()

print('\n训练完成！')


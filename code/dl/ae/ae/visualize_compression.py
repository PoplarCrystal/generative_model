"""
可视化脚本：展示MNIST图像从28×28压缩到16维潜在空间的过程
生成对比图：原始图片 | 28×28数字矩阵 | 16维编码向量
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchvision import datasets, transforms
from ae import AE

# 设置matplotlib
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# 设置参数
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
latent_dim = 16  # 使用16维潜在空间
input_dim = 784  # 28 * 28
hidden_dim = 128

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建模型（使用16维潜在空间）
model = AE(input_dim, hidden_dim, latent_dim).to(device)

# 尝试加载预训练模型
model_path = os.path.join(current_dir, 'ae_model.pth')
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'Loaded pretrained model: {model_path}')
    except:
        print('Failed to load model, using random initialization')
else:
    print('Model file not found, using random initialization')

# 选择一个样本（例如第一个数字'7'）
model.eval()
sample_image = None
sample_label = None

# 选择一个清晰的样本（比如第一个）
for x, label in test_loader:
    sample_image = x
    sample_label = label.item()
    break

# 将图像转换为numpy数组用于显示
img_np = sample_image.squeeze().numpy()  # 28×28
img_flat = sample_image.view(1, input_dim).to(device)

# 通过编码器获取潜在向量
with torch.no_grad():
    z = model.encoder(img_flat)  # 16维向量
    z_np = z.cpu().squeeze().numpy()

# 创建可视化 - 三个子图大小一致
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Left: Original MNIST image
axes[0].imshow(img_np, cmap='gray')
axes[0].set_title('Original MNIST Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Middle: 28×28 number matrix (display pixel values)
img_matrix = (img_np * 255).astype(int)
axes[1].axis('off')
axes[1].set_title('28×28 Pixel Matrix (0-255)', fontsize=12, fontweight='bold')
# 创建文本表格显示数字，使用更紧凑的格式
table_text = ''
for i in range(28):
    row = ' '.join([f'{img_matrix[i, j]:3d}' for j in range(28)])
    table_text += row + '\n'
axes[1].text(0.5, 0.5, table_text, transform=axes[1].transAxes,
            fontsize=7, family='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Right: 16-D latent vector (display 16 numbers)
axes[2].axis('off')
axes[2].set_title('16-D Latent Vector z', fontsize=12, fontweight='bold')
# 将16个数字格式化显示，每行4个数字，共4行
z_text_lines = []
for i in range(0, 16, 4):
    line = ' '.join([f'{z_np[j]:8.3f}' for j in range(i, min(i+4, 16))])
    z_text_lines.append(line)
z_text = '\n'.join(z_text_lines)
axes[2].text(0.5, 0.5, z_text, transform=axes[2].transAxes,
            fontsize=14, family='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()

# 保存图片
output_path = os.path.join(current_dir, 'mnist_compression_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Visualization saved: {output_path}')

plt.show()


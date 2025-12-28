import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import datasets, transforms
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

# 从配置文件读取模型文件路径
vqvae_model_name = config['model_file']['model_name']
pixelcnn_model_name = config['pixelcnn_model_file']['model_name']
# 模型保存路径：与代码文件路径一致
vqvae_model_path = os.path.join(current_dir, vqvae_model_name)
pixelcnn_model_path = os.path.join(current_dir, pixelcnn_model_name)

# 从配置文件读取生成参数
sample_size = config['visualization']['sample_size']
nrow = config['visualization']['nrow']
H_latent = config['latent_space']['H_latent']
W_latent = config['latent_space']['W_latent']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 第二步：加载模型
vqvae = VQVAE(input_dim, dim, n_embedding).to(device)
pixelcnn = PixelCNNWithEmbedding(
    n_blocks=pixelcnn_n_blocks,
    p=pixelcnn_dim,
    linear_dim=pixelcnn_linear_dim,
    bn=pixelcnn_use_bn,
    color_level=n_embedding
).to(device)

try:
    vqvae.load_state_dict(torch.load(vqvae_model_path, weights_only=True))
    print(f'VQ-VAE 模型已加载: {vqvae_model_path}')
    pixelcnn.load_state_dict(torch.load(pixelcnn_model_path, weights_only=True))
    print(f'PixelCNN 模型已加载: {pixelcnn_model_path}')
except FileNotFoundError as e:
    print(f'错误: 未找到模型文件: {e}')
    exit(1)

vqvae.eval()
pixelcnn.eval()

# 第三步：使用 PixelCNN 生成新的 codebook 索引
print('\n开始生成图像...')
print('注意: PixelCNN 是自回归模型，逐像素生成，速度较慢...')

generated_indices_list = []

with torch.no_grad():
    for sample_idx in range(sample_size):
        # 初始化：全零索引（或随机初始化）
        indices = torch.zeros((1, H_latent, W_latent), dtype=torch.long, device=device)
        
        # 自回归生成：逐像素预测
        for h in range(H_latent):
            for w in range(W_latent):
                # 获取当前预测
                logits = pixelcnn(indices)  # (1, n_embedding, H, W)
                # 获取当前位置的 logits
                probs = torch.softmax(logits[0, :, h, w], dim=0)  # (n_embedding,)
                # 采样
                sampled_idx = torch.multinomial(probs, 1).item()
                indices[0, h, w] = sampled_idx
        
        generated_indices_list.append(indices.cpu())
        
        if (sample_idx + 1) % 4 == 0:
            print(f'已生成 {sample_idx + 1}/{sample_size} 个样本')

# 拼接所有生成的索引
generated_indices = torch.cat(generated_indices_list, dim=0)  # (sample_size, H, W)
generated_indices = generated_indices.to(device)

# 第四步：使用 VQ-VAE 解码生成图像
print('\n使用 VQ-VAE 解码生成图像...')
with torch.no_grad():
    generated_images = vqvae.decode(generated_indices).cpu()  # (sample_size, 1, 28, 28)

# 第五步：可视化生成的图像
fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
fig.suptitle('Generated Images using PixelCNN + VQ-VAE', fontsize=16, y=0.995)

for idx, ax in enumerate(axes.flat):
    if idx < sample_size:
        ax.imshow(generated_images[idx].squeeze().numpy(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
plt.show()

print('\n生成完成！')


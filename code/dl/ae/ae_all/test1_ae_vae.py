"""
测试1: AE vs VAE 解码器对比
从标准正态分布采样latent z，分别使用AE和VAE的解码器进行解码，对比生成的图片
"""
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models import AE, VAE


# 设置参数
input_dim = 784  # 28 * 28
hidden_dim = 128
latent_dim = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 获取data文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_rel = '../../data'
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

# 加载模型
print("加载模型...")
ae_model = AE(input_dim, hidden_dim, latent_dim)
vae_model = VAE(input_dim, hidden_dim, latent_dim)

ae_model_path = 'ae_model.pth'
vae_model_path = 'vae_model.pth'

if os.path.exists(ae_model_path):
    ae_model.load_state_dict(torch.load(ae_model_path, weights_only=True))
    ae_model = ae_model.to(device)
    ae_model.eval()
    print(f"AE模型已加载: {ae_model_path}")
else:
    raise FileNotFoundError(f"AE模型文件不存在: {ae_model_path}")

if os.path.exists(vae_model_path):
    vae_model.load_state_dict(torch.load(vae_model_path, weights_only=True))
    vae_model = vae_model.to(device)
    vae_model.eval()
    print(f"VAE模型已加载: {vae_model_path}")
else:
    raise FileNotFoundError(f"VAE模型文件不存在: {vae_model_path}")

# 从标准正态分布采样2个latent z
print("\n从标准正态分布采样2个latent z...")
z_samples = torch.randn(2, latent_dim).to(device)
print(f"采样的z: {z_samples.cpu().numpy()}")

# 使用AE解码器解码
print("\n使用AE解码器解码...")
with torch.no_grad():
    ae_images = ae_model.decoder(z_samples)
    ae_images = ae_images.view(2, 28, 28).cpu().numpy()

# 使用VAE解码器解码
print("使用VAE解码器解码...")
with torch.no_grad():
    vae_images = vae_model.decoder(z_samples)
    vae_images = vae_images.view(2, 28, 28).cpu().numpy()

# 可视化对比
print("\n生成对比图...")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    # AE解码结果
    axes[i, 0].imshow(ae_images[i], cmap='gray')
    axes[i, 0].set_title(f'AE Decoder - Sample {i+1}\nz={z_samples[i].cpu().numpy()}', fontsize=12)
    axes[i, 0].axis('off')
    
    # VAE解码结果
    axes[i, 1].imshow(vae_images[i], cmap='gray')
    axes[i, 1].set_title(f'VAE Decoder - Sample {i+1}\nz={z_samples[i].cpu().numpy()}', fontsize=12)
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('test1_ae_vae_comparison.png', dpi=150, bbox_inches='tight')
print("对比图已保存: test1_ae_vae_comparison.png")

plt.show()


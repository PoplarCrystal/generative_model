"""
诊断 VAE 重构质量的脚本
用于检查 VAE 是否能够很好地重构图像，这是 LDM 质量的关键
"""
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
from vae import VAE

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
img_size = config['model']['img_size']
in_channel = config['model']['in_channel']
input_dim = img_size * img_size * in_channel
latent_dim = config['vae']['latent_dim']
hidden_dim = config['vae']['hidden_dim']
vae_model_name = config['model_file']['vae_model_name']

# 如果模型文件名为空，自动生成
if not vae_model_name or vae_model_name.strip() == '':
    vae_model_name = 'vae_model.pth'
vae_path = os.path.join(current_dir, vae_model_name)

data_dir_rel = config['data']['data_dir']
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')
print(f'潜在空间维度: {latent_dim}')
print(f'压缩比: {input_dim / latent_dim:.2f}x')

# 加载 VAE
vae = VAE(input_dim=input_dim, output_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
try:
    vae.load_state_dict(torch.load(vae_path, weights_only=True))
    print(f'VAE 已加载: {vae_path}')
except Exception as e:
    print(f'无法加载 VAE: {e}')
    print('请先训练 VAE')
    exit(1)

# 加载测试数据
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 测试重构质量
vae.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    x_flat = images.view(images.shape[0], -1).to(device)
    
    # 编码-解码
    z, mu, sigma = vae.encode(x_flat)
    x_recon = vae.decode(z)
    
    # 计算重构误差
    mse = F.mse_loss(x_recon, x_flat, reduction='mean')
    print(f'\n重构 MSE: {mse.item():.6f}')
    
    # 可视化
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle(f'VAE Reconstruction Quality (MSE: {mse.item():.4f}, Latent: {latent_dim})', fontsize=14)
    
    for i in range(8):
        # 原始图像
        axes[0, i].imshow(images[i].squeeze().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {labels[i].item()}', fontsize=8)
        axes[0, i].axis('off')
        
        # 重构图像
        recon_img = x_recon[i].view(1, img_size, img_size).cpu()
        axes[1, i].imshow(recon_img.squeeze().numpy(), cmap='gray')
        axes[1, i].set_title(f'Recon {labels[i].item()}', fontsize=8)
        axes[1, i].axis('off')
        
        # 误差图
        error = torch.abs(images[i].squeeze() - recon_img.squeeze())
        axes[2, i].imshow(error.numpy(), cmap='hot')
        axes[2, i].set_title(f'Error', fontsize=8)
        axes[2, i].axis('off')
        
        # 潜在向量统计
        z_i = z[i].cpu().numpy()
        axes[3, i].bar(range(len(z_i)), z_i)
        axes[3, i].set_title(f'Latent z[{i}]', fontsize=8)
        axes[3, i].set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.show()
    
    # 潜在向量统计信息
    print(f'\n潜在向量统计:')
    print(f'  Mean: {z.mean().item():.4f}')
    print(f'  Std: {z.std().item():.4f}')
    print(f'  Min: {z.min().item():.4f}')
    print(f'  Max: {z.max().item():.4f}')


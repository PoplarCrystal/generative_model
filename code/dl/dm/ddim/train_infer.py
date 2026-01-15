import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

# 添加 ddpm 目录到路径，以便导入 UNet 和 Diffuser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ddpm'))
from unet import UNet
from ddpm import Diffuser
from ddim import DDIMDiffuser

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
img_size = config['model']['img_size']
in_channel = config['model']['in_channel']
time_embed_dim = config['model']['time_embed_dim']
num_timesteps = config['diffusion']['num_timesteps']
beta_start = config['diffusion']['beta_start']
beta_end = config['diffusion']['beta_end']
ddim_steps = config['diffusion']['ddim_steps']
eta = config['diffusion']['eta']
infer = config['flags']['infer']
model_path_rel = config['model_file']['model_path']
sample_size = config['visualization']['sample_size']
nrow = config['visualization']['nrow']

# 获取模型文件的绝对路径
model_path = os.path.join(current_dir, model_path_rel)
model_path = os.path.normpath(model_path)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 模型构建（使用与 ddpm 相同的 UNet）
model = UNet(in_ch=in_channel, time_embed_dim=time_embed_dim).to(device)

# 加载 ddpm 的预训练模型
try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f'模型已加载，路径: {model_path}')
except FileNotFoundError:
    raise FileNotFoundError(f'未找到模型文件 {model_path}，请先训练 DDPM 模型或指定正确的模型路径')
except Exception as e:
    raise RuntimeError(f'加载模型时出错: {e}')

# 创建 DDPM 和 DDIM 采样器
ddpm_diffuser = Diffuser(num_timesteps=num_timesteps, beta_start=beta_start, 
                         beta_end=beta_end, device=device)
ddim_diffuser = DDIMDiffuser(num_timesteps=num_timesteps, beta_start=beta_start,
                             beta_end=beta_end, device=device, eta=eta)

# 模型推理 - 对比 DDPM 和 DDIM 采样
if infer:
    model.eval()
    print('\n开始生成图像对比...')
    
    x_shape = (sample_size, in_channel, img_size, img_size)
    
    # 使用相同的随机种子，确保对比公平
    torch.manual_seed(42)
    
    # DDPM 采样（1000步）
    print(f'\nDDPM 采样中（{num_timesteps} 步）...')
    ddpm_images = ddpm_diffuser.sample(model, x_shape)
    
    # DDIM 采样（50步）
    print(f'\nDDIM 采样中（{ddim_steps} 步）...')
    torch.manual_seed(42)  # 使用相同的随机种子
    ddim_images = ddim_diffuser.sample(model, x_shape, ddim_steps=ddim_steps)
    
    # 调试：检查图像值域
    if len(ddim_images) > 0:
        test_img = ddim_images[0]
        test_array = np.array(test_img.convert('L'))
        print(f'DDIM 图像值域: min={test_array.min()}, max={test_array.max()}, mean={test_array.mean():.2f}')
    
    # 将 PIL 图像转换为 numpy 数组用于显示
    ddpm_images_np = []
    for img in ddpm_images:
        img_np = np.array(img.convert('L'))
        ddpm_images_np.append(img_np)
    
    ddim_images_np = []
    for img in ddim_images:
        img_np = np.array(img.convert('L'))
        ddim_images_np.append(img_np)
    
    # 显示 DDPM 采样结果（1000步）
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle(f'DDPM Sampling ({num_timesteps} steps)', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(ddpm_images_np[idx], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()
    
    # 显示 DDIM 采样结果（50步）
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle(f'DDIM Sampling ({ddim_steps} steps, eta={eta})', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(ddim_images_np[idx], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()
    
    print(f'\n对比完成！')
    print(f'DDPM: {num_timesteps} 步采样')
    print(f'DDIM: {ddim_steps} 步采样（加速 {num_timesteps // ddim_steps}x）')


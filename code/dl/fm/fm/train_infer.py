"""
Flow Matching 训练与推理脚本。
接口与 ddpm3/train_infer.py 对齐：数据 [0,1]、config 驱动、训练/测试/推理/可视化。
"""
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import numpy as np
from unet import UNet
from fm import FlowMatcher

# 加载配置
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

img_size = config['model']['img_size']
in_channel = config['model']['in_channel']
time_embed_dim = config['model']['time_embed_dim']
fm_cfg = config['flow_matching']
time_embed_scale = fm_cfg['time_embed_scale']
num_sample_steps = fm_cfg['num_sample_steps']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
train = config['flags']['train']
infer = config['flags']['infer']
model_name = config['model_file'].get('model_name', 'flow_matching_model.pth')
model_path = os.path.join(current_dir, model_name)
data_dir_rel = config['data']['data_dir']
sample_size = config['visualization']['sample_size']
nrow = config['visualization']['nrow']

data_dir = os.path.normpath(os.path.join(current_dir, data_dir_rel))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 数据：与 DDPM 一致，ToTensor 得到 [0,1]
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型与 Flow Matcher
model = UNet(in_ch=in_channel, time_embed_dim=time_embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
flow_matcher = FlowMatcher(
    device=device,
    time_embed_scale=time_embed_scale,
    num_sample_steps=num_sample_steps,
)
print(f'Flow Matching: time_embed_scale={time_embed_scale}, num_sample_steps={num_sample_steps}')

train_losses = []


if train:
    print('\n开始训练...')
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            optimizer.zero_grad()
            x_1 = images.to(device)
            batch_size_b = x_1.size(0)
            x_0 = torch.randn_like(x_1, device=device)
            t = torch.rand(batch_size_b, device=device)
            x_t, target_velocity = flow_matcher.flow_forward(x_0, x_1, t)
            t_embed = flow_matcher.t_to_embed_index(t).long().clamp(1, time_embed_scale)
            pred_velocity = model(x_t, t_embed)
            loss = F.mse_loss(pred_velocity, target_velocity)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1
            pbar.set_postfix({'loss': f'{loss_sum/cnt:.4f}'})
        train_loss_avg = loss_sum / cnt
        train_losses.append(train_loss_avg)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}')

    torch.save(model.state_dict(), model_path)
    print(f'模型已保存: {model_path}')

    print('\n开始测试...')
    model.eval()
    test_loss_sum = 0.0
    test_cnt = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, _ in pbar:
            x_1 = images.to(device)
            batch_size_b = x_1.size(0)
            x_0 = torch.randn_like(x_1, device=device)
            t = torch.rand(batch_size_b, device=device)
            x_t, target_velocity = flow_matcher.flow_forward(x_0, x_1, t)
            t_embed = flow_matcher.t_to_embed_index(t).long().clamp(1, time_embed_scale)
            pred_velocity = model(x_t, t_embed)
            loss = F.mse_loss(pred_velocity, target_velocity)
            test_loss_sum += loss.item()
            test_cnt += 1
            pbar.set_postfix({'test_loss': f'{test_loss_sum/test_cnt:.4f}'})
    test_loss_avg = test_loss_sum / test_cnt
    print(f'Test Loss: {test_loss_avg:.4f}\n')

    if train_losses:
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Train Loss')
        plt.axhline(y=test_loss_avg, color='r', linestyle='--', label=f'Test Loss: {test_loss_avg:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Flow Matching Training Loss')
        plt.show()

if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载: {model_path}')
    except FileNotFoundError:
        raise FileNotFoundError(f'未找到模型文件 {model_path}，请先训练或指定正确路径')
    except Exception as e:
        raise RuntimeError(f'加载模型时出错: {e}')

if infer:
    model.eval()
    print('\n开始生成图像...')
    x_shape = (sample_size, in_channel, img_size, img_size)
    images = flow_matcher.sample(model, x_shape)
    images_np = [np.array(img.convert('L')) for img in images]

    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle('Generated Images (Flow Matching)', fontsize=16, y=0.995)
    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(images_np[idx], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()

    print('\n显示流过程（中间步）...')
    torch.manual_seed(42)
    noise = torch.randn(x_shape, device=device)
    step_interval = max(1, num_sample_steps // 16)
    _, intermediate_images, timestep_labels = flow_matcher.sample_with_intermediate_steps(
        model, x_shape=x_shape, step_interval=step_interval, noise=noise
    )
    intermediate_images_np = [np.array(img.convert('L')) for img in intermediate_images]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Flow Process (noise to image)', fontsize=16, y=0.995)
    for idx, ax in enumerate(axes.flat):
        if idx < len(intermediate_images_np):
            ax.imshow(intermediate_images_np[idx], cmap='gray')
            ax.set_title(timestep_labels[idx], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.1)
    plt.show()

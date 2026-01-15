import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import numpy as np
from unet import UNetCond
from cddpm import Diffuser

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
img_size = config['model']['img_size']
in_channel = config['model']['in_channel']
time_embed_dim = config['model']['time_embed_dim']
num_labels = config['model']['num_labels']
num_timesteps = config['diffusion']['num_timesteps']
beta_start = config['diffusion']['beta_start']
beta_end = config['diffusion']['beta_end']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
train = config['flags']['train']
infer = config['flags']['infer']
model_name = config['model_file']['model_name']
# 如果 model_name 为空，自动生成
if not model_name or model_name.strip() == '':
    model_name = 'cddpm_model.pth'
    print(f'模型文件名未指定，自动生成: {model_name}')
else:
    print(f'使用指定的模型文件名: {model_name}')
# 模型保存路径：与代码文件路径一致
model_path = os.path.join(current_dir, model_name)
data_dir_rel = config['data']['data_dir']
sample_size = config['visualization']['sample_size']
nrow = config['visualization']['nrow']

# 获取data文件夹的绝对路径
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型构建
model = UNetCond(in_ch=in_channel, time_embed_dim=time_embed_dim, num_labels=num_labels).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'使用学习率: {learning_rate}')
print(f'标签数量: {num_labels}')

# 扩散过程管理器
diffuser = Diffuser(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, device=device)

train_losses = []


def show_images(images, labels=None, rows=2, cols=10):
    """显示图像网格（支持标签显示）"""
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i < len(images):
                ax = fig.add_subplot(rows, cols, i + 1)
                plt.imshow(images[i], cmap='gray')
                if labels is not None:
                    ax.set_xlabel(f'Label: {labels[i].item()}', fontsize=8)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                i += 1
    plt.tight_layout()
    plt.show()


if train:
    # 模型训练
    print('\n开始训练...')
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            optimizer.zero_grad()
            x = images.to(device)
            labels = labels.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.diffusion(x, t)
            noise_pred = model(x_noisy, t, labels)  # 传入labels进行条件生成
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
            pbar.set_postfix({'loss': f'{loss_sum/cnt:.4f}'})

        train_loss_avg = loss_sum / cnt
        train_losses.append(train_loss_avg)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存，路径: {model_path}')

    # 模型测试
    print('\n开始测试...')
    model.eval()
    test_loss_sum = 0.0
    test_cnt = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            x = images.to(device)
            labels = labels.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.diffusion(x, t)
            noise_pred = model(x_noisy, t, labels)
            loss = F.mse_loss(noise, noise_pred)

            test_loss_sum += loss.item()
            test_cnt += 1
            pbar.set_postfix({'test_loss': f'{test_loss_sum/test_cnt:.4f}'})

    test_loss_avg = test_loss_sum / test_cnt
    print(f'Test Loss: {test_loss_avg:.4f}\n')

    # 模型可视化 - 训练损失曲线
    if len(train_losses) > 0:
        epochs_list = list(range(1, len(train_losses) + 1))
        plt.plot(epochs_list, train_losses, marker='o', linestyle='-', label='Train Loss')
        plt.axhline(y=test_loss_avg, color='r', linestyle='--', label=f'Test Loss: {test_loss_avg:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('CDDPM Training Loss')
        plt.show()
    else:
        print(f'测试损失: {test_loss_avg:.4f} (未训练，无法显示训练损失曲线)')

# 加载模型（如果不需要训练，也需要加载模型用于推理）
if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载，路径: {model_path}')
    except FileNotFoundError:
        raise FileNotFoundError(f'未找到模型文件 {model_path}，请先训练模型或指定正确的模型路径')
    except Exception as e:
        raise RuntimeError(f'加载模型时出错: {e}')

# 模型推理 - 生成新图像（条件生成）
if infer:
    model.eval()
    print('\n开始生成图像...')
    
    # 生成图像（可以指定标签或随机生成）
    x_shape = (sample_size, in_channel, img_size, img_size)
    # 可以指定特定标签，例如生成数字0-9各一张
    # labels = torch.tensor([i for i in range(10)] + [i for i in range(6)], device=device)
    labels = None  # None表示随机生成标签
    images, labels = diffuser.sample(model, x_shape, labels=labels)
    
    # 将最终生成的 PIL 图像转换为 numpy 数组用于显示
    images_np = []
    for img in images:
        img_np = np.array(img.convert('L'))  # 转换为灰度图
        images_np.append(img_np)
    
    # 显示最终生成的图像
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle('Generated Images (CDDPM - Conditional DDPM)', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(images_np[idx], cmap='gray')
            ax.set_title(f'Label: {labels[idx].item()}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import numpy as np
from gan import Generator, Discriminator

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
img_size = config['model']['img_size']
in_channel = config['model']['in_channel']
noise_dim = config['model']['noise_dim']
hidden_dim = config['model']['hidden_dim']
input_dim = img_size * img_size * in_channel  # MNIST: 28*28*1 = 784
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
n_critic = config['training']['n_critic']
train = config['flags']['train']
infer = config['flags']['infer']
generator_name = config['model_file']['generator_name']
discriminator_name = config['model_file']['discriminator_name']

# 如果模型文件名为空，自动生成
if not generator_name or generator_name.strip() == '':
    generator_name = 'generator_model.pth'
    print(f'生成器文件名未指定，自动生成: {generator_name}')
else:
    print(f'使用指定的生成器文件名: {generator_name}')

if not discriminator_name or discriminator_name.strip() == '':
    discriminator_name = 'discriminator_model.pth'
    print(f'判别器文件名未指定，自动生成: {discriminator_name}')
else:
    print(f'使用指定的判别器文件名: {discriminator_name}')

# 模型保存路径
generator_path = os.path.join(current_dir, generator_name)
discriminator_path = os.path.join(current_dir, discriminator_name)
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型构建
generator = Generator(noise_dim=noise_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
discriminator = Discriminator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

# 优化器
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

print(f'使用学习率: {learning_rate}')
print(f'噪声维度: {noise_dim}')

# 损失函数
criterion = nn.BCELoss()

# 训练历史
g_losses = []
d_losses = []


def show_images(images, rows=2, cols=10):
    """显示图像网格"""
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i < len(images):
                fig.add_subplot(rows, cols, i + 1)
                plt.imshow(images[i], cmap='gray')
                plt.axis('off')
                i += 1
    plt.tight_layout()
    plt.show()


if train:
    # 模型训练
    print('\n开始训练 GAN...')
    for epoch in range(epochs):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            batch_size_actual = images.size(0)
            images = images.view(batch_size_actual, -1).to(device)  # [B, 784]
            
            # 真实标签和虚假标签
            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)

            # ========== 训练判别器 ==========
            d_optimizer.zero_grad()
            
            # 真实图像的损失
            real_output = discriminator(images)
            d_loss_real = criterion(real_output, real_labels)
            
            # 生成虚假图像
            noise = torch.randn(batch_size_actual, noise_dim, device=device)
            fake_images = generator(noise)
            
            # 虚假图像的损失
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ========== 训练生成器 ==========
            if (cnt + 1) % n_critic == 0:
                g_optimizer.zero_grad()
                
                # 生成虚假图像
                noise = torch.randn(batch_size_actual, noise_dim, device=device)
                fake_images = generator(noise)
                
                # 生成器希望判别器认为虚假图像是真实的
                fake_output = discriminator(fake_images)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_loss_sum += g_loss.item()
            else:
                g_loss_sum += 0.0
            
            d_loss_sum += d_loss.item()
            cnt += 1
            
            pbar.set_postfix({
                'd_loss': f'{d_loss_sum/cnt:.4f}',
                'g_loss': f'{g_loss_sum/max(1, (cnt//n_critic)):.4f}'
            })

        g_loss_avg = g_loss_sum / max(1, (cnt // n_critic))
        d_loss_avg = d_loss_sum / cnt
        g_losses.append(g_loss_avg)
        d_losses.append(d_loss_avg)
        print(f'Epoch {epoch+1}/{epochs}, D Loss: {d_loss_avg:.4f}, G Loss: {g_loss_avg:.4f}')

    # 保存模型
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f'生成器已保存，路径: {generator_path}')
    print(f'判别器已保存，路径: {discriminator_path}')

    # 训练损失曲线
    if len(g_losses) > 0:
        epochs_list = list(range(1, len(g_losses) + 1))
        plt.plot(epochs_list, g_losses, label='Generator Loss', marker='o')
        plt.plot(epochs_list, d_losses, label='Discriminator Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Loss')
        plt.show()

# 加载模型（如果不需要训练，也需要加载模型用于推理）
if not train:
    try:
        generator.load_state_dict(torch.load(generator_path, weights_only=True))
        print(f'生成器已加载，路径: {generator_path}')
    except FileNotFoundError:
        raise FileNotFoundError(
            f'未找到生成器文件 {generator_path}，请先训练模型或指定正确的模型路径'
        )
    except Exception as e:
        raise RuntimeError(f'加载生成器时出错: {e}')
    
    try:
        discriminator.load_state_dict(torch.load(discriminator_path, weights_only=True))
        print(f'判别器已加载，路径: {discriminator_path}')
    except FileNotFoundError:
        raise FileNotFoundError(
            f'未找到判别器文件 {discriminator_path}，请先训练模型或指定正确的模型路径'
        )
    except Exception as e:
        raise RuntimeError(f'加载判别器时出错: {e}')

# 模型推理 - 生成新图像
if infer:
    generator.eval()
    print('\n开始生成图像...')
    
    # 生成图像
    with torch.no_grad():
        noise = torch.randn(sample_size, noise_dim, device=device)
        fake_images = generator(noise)
        fake_images = fake_images.view(sample_size, in_channel, img_size, img_size)
    
    # 将张量转换为 numpy 数组用于显示
    # 确保值在[0,1]范围内
    fake_images = torch.clamp(fake_images, 0.0, 1.0)
    images_np = []
    for i in range(sample_size):
        img = fake_images[i].cpu().numpy().squeeze()
        images_np.append(img)
    
    # 显示生成的图像
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle('Generated Images (GAN)', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(images_np[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()
    
    # 噪声可视化：对生成的图像添加不同比例的噪声，并用判别器判断
    print('\n显示噪声对判别器判断的影响...')
    generator.eval()
    discriminator.eval()
    
    # 生成一张新图像
    with torch.no_grad():
        noise = torch.randn(1, noise_dim, device=device)
        original_image = generator(noise)  # [1, 784]
        original_image_2d = original_image.view(1, in_channel, img_size, img_size)
    
    # 创建不同噪声比例的图像
    num_alphas = 16
    alphas = [i / num_alphas for i in range(num_alphas + 1)]  # [0, 1/16, 2/16, ..., 1]
    noisy_images = []
    predictions = []
    alpha_labels = []
    
    with torch.no_grad():
        for alpha in alphas:
            # 添加噪声：x = alpha * noise + (1 - alpha) * x
            noise_tensor = torch.randn_like(original_image)
            noisy_image = alpha * noise_tensor + (1 - alpha) * original_image
            noisy_images.append(noisy_image)
            
            # 用判别器判断
            pred = discriminator(noisy_image)
            predictions.append(pred.item())
            alpha_labels.append(f'α={alpha:.2f}, D={pred.item():.3f}')
    
    # 转换为 numpy 数组用于显示
    # 确保值在[0,1]范围内
    images_np = []
    for img in noisy_images:
        img = torch.clamp(img, 0.0, 1.0)
        img_2d = img.view(1, in_channel, img_size, img_size)
        images_np.append(img_2d[0].cpu().numpy().squeeze())
    
    # 显示结果
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Noise Effect on Discriminator (α: noise ratio)', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images_np):
            ax.imshow(images_np[idx], cmap='gray')
            # 判断是真还是假（阈值 0.5）
            is_real = 'Real' if predictions[idx] > 0.5 else 'Fake'
            ax.set_title(f'{alpha_labels[idx]}\n{is_real}', fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3, wspace=0.1)
    plt.show()


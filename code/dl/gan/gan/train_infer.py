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
# 归一化到 [-1, 1] 以匹配生成器的 Tanh 输出
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]
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
            images = images.view(-1, input_dim).to(device)  # [B, 784]，-1 自动推断batch大小
            
            # 真实标签和虚假标签
            real_labels = torch.ones(len(images), 1, device=device)
            fake_labels = torch.zeros(len(images), 1, device=device)

            # ========== 训练判别器 n_critic 次 ==========
            # 让判别器训练更充分，通常 n_critic=5 效果较好
            d_loss_batch = 0.0
            for _ in range(n_critic):
                d_optimizer.zero_grad()
                
                # 真实图像的损失
                real_output = discriminator(images)
                d_loss_real = criterion(real_output, real_labels)
                
                # 生成虚假图像（每次都用新的噪声）
                noise = torch.randn(len(images), noise_dim, device=device)
                fake_images = generator(noise)
                
                # 虚假图像的损失
                fake_output = discriminator(fake_images.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                d_loss_batch += d_loss.item()
            
            d_loss_avg = d_loss_batch / n_critic
            d_loss_sum += d_loss_avg
            
            # ========== 训练生成器 1 次 ==========
            g_optimizer.zero_grad()
            
            # 生成虚假图像
            noise = torch.randn(len(images), noise_dim, device=device)
            fake_images = generator(noise)
            
            # 生成器希望判别器认为虚假图像是真实的
            fake_output = discriminator(fake_images)
            # 使用非饱和 loss：-log(D(x_fake))，而不是 BCE loss
            # 这样可以避免梯度消失问题，训练更稳定
            # 原 BCE loss: g_loss = criterion(fake_output, real_labels)
            # 非饱和 loss 等价于最大化 log(D(x_fake))，即最小化 -log(D(x_fake))
            # 为了数值稳定性，对 fake_output 进行 clamp，避免 log(0)
            fake_output_clamped = torch.clamp(fake_output, min=1e-8, max=1.0)
            g_loss = -torch.mean(torch.log(fake_output_clamped))
            
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_sum += g_loss.item()
            cnt += 1
            
            pbar.set_postfix({
                'd_loss': f'{d_loss_sum/cnt:.4f}',
                'g_loss': f'{g_loss_sum/cnt:.4f}'
            })

        g_loss_avg = g_loss_sum / cnt
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
    discriminator.eval()
    print('\n开始生成图像和可视化...')
    
    # ========== 图1: 16张真实图像及其判别器输出 ==========
    print('\n生成图1: 16张真实图像的判别器判断...')
    real_images_list = []
    real_predictions = []
    
    with torch.no_grad():
        # 收集16张真实图像
        count = 0
        for images, _ in train_loader:
            images_flat = images.view(images.size(0), -1).to(device)  # [B, 784]
            preds = discriminator(images_flat)
            
            for i in range(len(images)):
                if count >= 16:
                    break
                real_images_list.append(images[i].cpu())  # [1, 28, 28]
                real_predictions.append(preds[i].item())
                count += 1
            if count >= 16:
                break
    
    # 转换为numpy用于显示（真实图像已经是[0,1]范围，因为transform已经处理过）
    real_images_np = []
    for img in real_images_list:
        img_np = img.numpy().squeeze()
        # 如果图像是[-1,1]范围，需要转换
        if img_np.min() < 0:
            img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0.0, 1.0)
        real_images_np.append(img_np)
    
    # 显示真实图像
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Real Images - Discriminator Predictions', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(real_images_np):
            ax.imshow(real_images_np[idx], cmap='gray')
            pred_value = real_predictions[idx]
            is_real = 'Real' if pred_value > 0.5 else 'Fake'
            color = 'green' if pred_value > 0.5 else 'red'
            ax.set_title(f'D={pred_value:.3f}\n{is_real}', 
                        fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.1)
    plt.show()
    
    # ========== 图2: 16张生成图像及其判别器输出 ==========
    print('\n生成图2: 16张生成图像的判别器判断...')
    fake_images_list = []
    fake_predictions = []
    
    with torch.no_grad():
        # 生成16张图像
        noise = torch.randn(16, noise_dim, device=device)
        fake_images_flat = generator(noise)  # [16, 784]
        fake_preds = discriminator(fake_images_flat)
        
        # 转换为图像格式 [16, 1, 28, 28]
        fake_images_2d = fake_images_flat.view(16, in_channel, img_size, img_size)
        
        for i in range(16):
            fake_images_list.append(fake_images_2d[i].cpu())
            fake_predictions.append(fake_preds[i].item())
    
    # 转换为numpy用于显示（生成器输出是[-1,1]，需要转换）
    fake_images_np = []
    for img in fake_images_list:
        img_np = img.numpy().squeeze()
        img_np = (img_np + 1) / 2  # [-1, 1] -> [0, 1]
        img_np = np.clip(img_np, 0.0, 1.0)
        fake_images_np.append(img_np)
    
    # 显示生成图像
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Generated Images - Discriminator Predictions', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(fake_images_np):
            ax.imshow(fake_images_np[idx], cmap='gray')
            pred_value = fake_predictions[idx]
            is_real = 'Real' if pred_value > 0.5 else 'Fake'
            color = 'green' if pred_value > 0.5 else 'red'
            ax.set_title(f'D={pred_value:.3f}\n{is_real}', 
                        fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.1)
    plt.show()
    
    # ========== 图3: 真实图像和生成图像的混合 ==========
    print('\n生成图3: 真实图像和生成图像的混合效果...')
    
    # 获取一张真实图像
    real_image, _ = next(iter(train_loader))
    real_image_flat = real_image[0:1].view(1, -1).to(device)  # [1, 784]
    
    # 生成一张新图像
    with torch.no_grad():
        noise = torch.randn(1, noise_dim, device=device)
        fake_image_flat = generator(noise)  # [1, 784]
    
    # 创建不同混合比例的图像（alpha: 真实图像的比例）
    num_alphas = 16
    alphas = [i / num_alphas for i in range(num_alphas + 1)]  # [0, 1/16, 2/16, ..., 1]
    mixed_images = []
    predictions = []
    alpha_labels = []
    
    with torch.no_grad():
        for alpha in alphas:
            # 混合：x = alpha * real + (1 - alpha) * fake
            mixed_image = alpha * real_image_flat + (1 - alpha) * fake_image_flat
            mixed_images.append(mixed_image)
            
            # 用判别器判断
            pred = discriminator(mixed_image)
            predictions.append(pred.item())
            alpha_labels.append(f'α={alpha:.2f}\nD={pred.item():.3f}')
    
    # 转换为 numpy 数组用于显示
    # 图像是 [-1, 1] 范围，需要转换回 [0, 1] 用于显示
    mixed_images_np = []
    for img in mixed_images:
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = torch.clamp(img, 0.0, 1.0)
        img_2d = img.view(1, in_channel, img_size, img_size)
        mixed_images_np.append(img_2d[0].cpu().numpy().squeeze())
    
    # 显示混合结果
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Real vs Fake Mixing Effect (α: real image ratio)', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(mixed_images_np):
            ax.imshow(mixed_images_np[idx], cmap='gray')
            pred_value = predictions[idx]
            is_real = 'Real' if pred_value > 0.5 else 'Fake'
            color = 'green' if pred_value > 0.5 else 'red'
            ax.set_title(f'{alpha_labels[idx]}\n{is_real}', 
                        fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3, wspace=0.1)
    plt.show()
    
    # 打印统计信息
    print('\n判别器统计信息:')
    real_mean = np.mean(real_predictions)
    fake_mean = np.mean(fake_predictions)
    print(f'真实图像平均输出: {real_mean:.4f}')
    print(f'生成图像平均输出: {fake_mean:.4f}')
    
    # GAN的最终目标（纳什均衡）：判别器无法区分真实和生成图像
    # 理想情况下，两者都应该接近 0.5（随机猜测）
    print('\nGAN训练目标说明:')
    print('  - 训练过程中: 真实图像输出 > 0.5，生成图像输出 < 0.5')
    print('  - 理想状态（纳什均衡）: 两者都接近 0.5（判别器无法区分）')
    
    # 判断训练状态
    if real_mean < 0.5:
        print('\n⚠️  警告: 判别器对真实图像的输出 < 0.5')
        print('   可能原因: 判别器训练不足，或生成器过强导致判别器被"欺骗"')
        print('   建议: 检查训练过程，可能需要调整 n_critic 或学习率')
    elif abs(real_mean - 0.5) < 0.1 and abs(fake_mean - 0.5) < 0.1:
        print('\n✅ 训练状态良好: 判别器输出接近 0.5，说明生成器已能生成逼真图像！')
        print('   (这是GAN的理想状态：判别器无法区分真实和生成图像)')
    elif real_mean > 0.7 and fake_mean < 0.3:
        print('\n📊 训练进行中: 判别器仍能较好地区分真实和生成图像')
        print('   建议: 继续训练以达到纳什均衡（两者都接近 0.5）')
    else:
        print('\n📊 当前训练状态: 需要继续训练以达到平衡')


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import numpy as np
from wgan import Generator, Critic

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
input_dim = img_size * img_size * in_channel
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
n_critic = config['training']['n_critic']
lambda_gp = config['training'].get('lambda_gp', 10)  # WGAN-GP 梯度惩罚系数
optimizer_name = config['training']['optimizer']
train = config['flags']['train']
infer = config['flags']['infer']
generator_name = config['model_file']['generator_name']
critic_name = config['model_file'].get('critic_name', config['model_file'].get('discriminator_name', ''))

# 如果模型文件名为空，自动生成
if not generator_name or generator_name.strip() == '':
    generator_name = 'wgan_gp_generator.pth'
if not critic_name or critic_name.strip() == '':
    critic_name = 'wgan_gp_critic.pth'

# 模型保存路径
generator_path = os.path.join(current_dir, generator_name)
critic_path = os.path.join(current_dir, critic_name)
data_dir_rel = config['data']['data_dir']

# 获取data文件夹的绝对路径
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]
])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 模型构建
generator = Generator(noise_dim=noise_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
critic = Critic(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

# 优化器 (WGAN-GP 推荐使用 Adam, betas=(0.0, 0.9))
if optimizer_name.lower() == 'rmsprop':
    g_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    c_optimizer = optim.RMSprop(critic.parameters(), lr=learning_rate)
else:
    # WGAN-GP 推荐 beta1=0.0, beta2=0.9
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    c_optimizer = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

print(f'使用优化器: {optimizer_name}')
print(f'Gradient Penalty Lambda: {lambda_gp}')

# 梯度惩罚计算函数
def compute_gradient_penalty(critic, real_samples, fake_samples):
    """计算 WGAN-GP 的梯度惩罚项"""
    # 随机权重项 epsilon
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    # 获取随机插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    # 获取梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# 训练历史
g_losses = []
c_losses = []

if train:
    # 模型训练
    print('\n开始训练 WGAN-GP...')
    for epoch in range(epochs):
        g_loss_sum = 0.0
        c_loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, (images, _) in enumerate(pbar):
            images = images.view(-1, input_dim).to(device)
            batch_size_curr = images.size(0)

            # ==================================================================
            # 1. 训练 Critic (n_critic 次)
            # ==================================================================
            
            c_optimizer.zero_grad()
            
            # 1.1 真实图像 Loss
            real_validity = critic(images)
            c_loss_real = -torch.mean(real_validity)
            
            # 1.2 生成图像 Loss
            noise = torch.randn(batch_size_curr, noise_dim, device=device)
            fake_images = generator(noise).detach()
            fake_validity = critic(fake_images)
            c_loss_fake = torch.mean(fake_validity)
            
            # 1.3 梯度惩罚 (Gradient Penalty)
            gradient_penalty = compute_gradient_penalty(critic, images.data, fake_images.data)
            
            # Critic 总 Loss
            c_loss = c_loss_real + c_loss_fake + gradient_penalty
            c_loss.backward()
            c_optimizer.step()
            
            c_loss_sum += c_loss.item()

            # ==================================================================
            # 2. 训练 Generator (每 n_critic 步训练一次)
            # ==================================================================
            if (i + 1) % n_critic == 0:
                g_optimizer.zero_grad()
                
                # 生成新图像
                noise = torch.randn(batch_size_curr, noise_dim, device=device)
                fake_images = generator(noise)
                
                # Generator Loss
                fake_validity = critic(fake_images)
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_loss_sum += g_loss.item()
                cnt += 1
                
                pbar.set_postfix({
                    'c_loss': f'{c_loss.item():.4f}',
                    'g_loss': f'{g_loss.item():.4f}',
                    'gp': f'{gradient_penalty.item():.4f}'
                })

        # 记录平均 Loss
        if cnt > 0:
            g_loss_avg = g_loss_sum / cnt
            c_loss_avg = c_loss_sum / len(train_loader)
            g_losses.append(g_loss_avg)
            c_losses.append(c_loss_avg)
            print(f'Epoch {epoch+1}/{epochs}, Critic Loss: {c_loss_avg:.4f}, Generator Loss: {g_loss_avg:.4f}')

    # 保存模型
    torch.save(generator.state_dict(), generator_path)
    torch.save(critic.state_dict(), critic_path)
    print(f'生成器已保存: {generator_path}')
    print(f'Critic已保存: {critic_path}')

    # 绘制 Loss 曲线
    if len(g_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(c_losses, label='Critic Loss')
        plt.title('WGAN-GP Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

# 加载模型用于推理
if not train:
    try:
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        print(f'生成器已加载，路径: {generator_path}')
    except FileNotFoundError:
        print(f'未找到生成器文件 {generator_path}，请先训练模型或指定正确的模型路径')
        if infer: exit()
    except Exception as e:
        raise RuntimeError(f'加载生成器时出错: {e}')

    try:
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        print(f'Critic 已加载，路径: {critic_path}')
    except FileNotFoundError:
        print(f'未找到 Critic 文件 {critic_path}，请先训练模型或指定正确的模型路径')
    except Exception as e:
        raise RuntimeError(f'加载 Critic 时出错: {e}')

# 模型推理与可视化
if infer:
    generator.eval()
    critic.eval()
    print('\n开始生成图像和可视化...')

    # ========== 图1: 16 张真实图像及其 Critic 分数 ==========
    print('\n生成图1: 16 张真实图像的 Critic 分数...')
    real_images_list = []
    real_scores = []

    with torch.no_grad():
        count = 0
        for images, _ in train_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            scores = critic(images_flat)
            for i in range(len(images)):
                if count >= 16:
                    break
                real_images_list.append(images[i].cpu())
                real_scores.append(scores[i].item())
                count += 1
            if count >= 16:
                break

    real_images_np = []
    for img in real_images_list:
        img_np = img.numpy().squeeze()
        if img_np.min() < 0:
            img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0.0, 1.0)
        real_images_np.append(img_np)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Real Images - WGAN Critic Scores (higher = more real-like)', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < len(real_images_np):
            ax.imshow(real_images_np[idx], cmap='gray')
            score = real_scores[idx]
            ax.set_title(f'Score={score:.3f}', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.1)
    plt.show()

    # ========== 图2: 16 张生成图像及其 Critic 分数 ==========
    print('\n生成图2: 16 张生成图像的 Critic 分数...')
    fake_images_list = []
    fake_scores = []

    with torch.no_grad():
        noise = torch.randn(16, noise_dim, device=device)
        fake_images_flat = generator(noise)
        scores = critic(fake_images_flat)
        fake_images_2d = fake_images_flat.view(16, in_channel, img_size, img_size)
        for i in range(16):
            fake_images_list.append(fake_images_2d[i].cpu())
            fake_scores.append(scores[i].item())

    fake_images_np = []
    for img in fake_images_list:
        img_np = img.numpy().squeeze()
        img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0.0, 1.0)
        fake_images_np.append(img_np)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Generated Images - WGAN Critic Scores (higher = more real-like)', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < len(fake_images_np):
            ax.imshow(fake_images_np[idx], cmap='gray')
            score = fake_scores[idx]
            ax.set_title(f'Score={score:.3f}', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.1)
    plt.show()

    # ========== 图3: 真实与生成图像的混合 ==========
    print('\n生成图3: 真实图像和生成图像的混合效果...')

    real_image, _ = next(iter(train_loader))
    real_image_flat = real_image[0:1].view(1, -1).to(device)

    with torch.no_grad():
        noise = torch.randn(1, noise_dim, device=device)
        fake_image_flat = generator(noise)

    num_alphas = 16
    alphas = [i / num_alphas for i in range(num_alphas + 1)]
    mixed_images = []
    predictions = []
    alpha_labels = []

    with torch.no_grad():
        for alpha in alphas:
            mixed_image = alpha * real_image_flat + (1 - alpha) * fake_image_flat
            mixed_images.append(mixed_image)
            pred = critic(mixed_image)
            predictions.append(pred.item())
            alpha_labels.append(f'α={alpha:.2f}\nS={pred.item():.3f}')

    mixed_images_np = []
    for img in mixed_images:
        img = (img + 1) / 2
        img = torch.clamp(img, 0.0, 1.0)
        img_2d = img.view(1, in_channel, img_size, img_size)
        mixed_images_np.append(img_2d[0].cpu().numpy().squeeze())

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Real vs Fake Mixing (α: real ratio) - WGAN Critic Score', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < len(mixed_images_np):
            ax.imshow(mixed_images_np[idx], cmap='gray')
            ax.set_title(alpha_labels[idx], fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3, wspace=0.1)
    plt.show()

    # 统计信息
    print('\nCritic 分数统计:')
    real_mean = np.mean(real_scores)
    fake_mean = np.mean(fake_scores)
    print(f'真实图像平均分数: {real_mean:.4f}')
    print(f'生成图像平均分数: {fake_mean:.4f}')
    print('\nWGAN 说明: Critic 输出为标量分数，分数越高表示越像真实分布。')
    print('训练良好时，真实样本分数较高、生成样本分数逐步升高。')

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
from ldm import LatentDiffuser
from vae import VAE

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
input_dim = img_size * img_size * in_channel  # MNIST: 28*28*1 = 784
latent_dim = config['vae']['latent_dim']
hidden_dim = config['vae']['hidden_dim']
num_timesteps = config['diffusion']['num_timesteps']
beta_start = config['diffusion']['beta_start']
beta_end = config['diffusion']['beta_end']
# VAE 训练参数
vae_epochs = config['train_vae']['epochs']
vae_learning_rate = config['train_vae']['learning_rate']
vae_batch_size = config['train_vae']['batch_size']
# LDM 训练参数
ldm_epochs = config['train_ldm']['epochs']
ldm_learning_rate = config['train_ldm']['learning_rate']
ldm_batch_size = config['train_ldm']['batch_size']
cond_dropout_prob = config['train_ldm']['cond_dropout_prob']
# 执行控制标志
train_vae = config['flags']['train_vae']
train_ldm = config['flags']['train_ldm']
infer = config['flags']['infer']
vae_model_name = config['model_file']['vae_model_name']
model_name = config['model_file']['model_name']
# 如果 vae_model_name 为空，自动生成
if not vae_model_name or vae_model_name.strip() == '':
    vae_model_name = 'vae_model.pth'
    print(f'VAE 模型文件名未指定，自动生成: {vae_model_name}')
else:
    print(f'使用指定的 VAE 模型文件名: {vae_model_name}')
# 如果 model_name 为空，自动生成
if not model_name or model_name.strip() == '':
    model_name = 'ldm_model.pth'
    print(f'LDM 模型文件名未指定，自动生成: {model_name}')
else:
    print(f'使用指定的 LDM 模型文件名: {model_name}')
# 模型保存路径：与代码文件路径一致
model_path = os.path.join(current_dir, model_name)
vae_path = os.path.join(current_dir, vae_model_name)
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
# 使用 LDM 的 batch_size（因为 VAE 训练会单独处理）
train_loader = DataLoader(train_dataset, batch_size=ldm_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=ldm_batch_size, shuffle=False)

# VAE 模型构建
vae = VAE(
    input_dim=input_dim,
    output_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim
).to(device)
vae_optimizer = optim.Adam(vae.parameters(), lr=vae_learning_rate)

# 加载或训练 VAE
if train_vae:
    print('\n开始训练 VAE...')
    # VAE 训练时使用 VAE 的 batch_size
    vae_train_loader = DataLoader(train_dataset, batch_size=vae_batch_size, shuffle=True)
    for epoch in range(vae_epochs):
        loss_sum = 0.0
        cnt = 0
        pbar = tqdm(vae_train_loader, desc=f'VAE Epoch {epoch+1}/{vae_epochs}')
        for images, _ in pbar:
            vae_optimizer.zero_grad()
            x = images.to(device)
            x_flat = x.view(x.shape[0], -1)  # [B, C, H, W] -> [B, C*H*W]
            
            x_hat, mu, sigma = vae(x_flat)
            # VAE 损失：重构损失 + KL 散度
            recon_loss = F.mse_loss(x_hat, x_flat, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
            loss = (recon_loss + kl_loss) / x.shape[0]
            
            loss.backward()
            vae_optimizer.step()
            
            loss_sum += loss.item()
            cnt += 1
            pbar.set_postfix({'loss': f'{loss_sum/cnt:.4f}'})
        print(f'VAE Epoch {epoch+1}, Loss: {loss_sum/cnt:.4f}')
    torch.save(vae.state_dict(), vae_path)
    print(f'VAE 已保存，路径: {vae_path}')
else:
    try:
        vae.load_state_dict(torch.load(vae_path, weights_only=True))
        print(f'VAE 已加载，路径: {vae_path}')
    except FileNotFoundError:
        raise FileNotFoundError(f'未找到 VAE 文件 {vae_path}，请先训练 VAE 或指定正确的模型路径')
    except Exception as e:
        raise RuntimeError(f'加载 VAE 时出错: {e}')

# 冻结 VAE（只训练扩散模型）
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

# 扩散模型构建（在潜在空间）
model = UNetCond(
    latent_dim=latent_dim,
    feature_size=config['model'].get('feature_size', 8),
    feature_channels=config['model'].get('feature_channels', 4),
    time_embed_dim=time_embed_dim,
    num_labels=num_labels
).to(device)
optimizer = optim.Adam(model.parameters(), lr=ldm_learning_rate)
print(f'LDM 使用学习率: {ldm_learning_rate}')
print(f'标签数量: {num_labels}')
print(f'潜在空间维度: {latent_dim}')

# 潜在空间扩散过程管理器
diffuser = LatentDiffuser(
    num_timesteps=num_timesteps,
    beta_start=beta_start,
    beta_end=beta_end,
    device=device
)

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


if train_ldm:
    # 模型训练
    print('\n开始训练 LDM...')
    for epoch in range(ldm_epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{ldm_epochs}')
        for images, labels in pbar:
            optimizer.zero_grad()
            x = images.to(device)
            labels = labels.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            # CFG 训练：随机丢弃条件
            if np.random.random() < cond_dropout_prob:
                labels = None

            # 将图像编码到潜在空间
            x_flat = x.view(x.shape[0], -1)  # [B, C, H, W] -> [B, C*H*W]
            with torch.no_grad():
                z_0, _, _ = vae.encode(x_flat)  # [B, latent_dim]
            
            # 在潜在空间进行扩散
            z_noisy, noise = diffuser.diffusion(z_0, t)
            noise_pred = model(z_noisy, t, labels)  # 传入labels（可能为None）
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
            pbar.set_postfix({'loss': f'{loss_sum/cnt:.4f}'})

        train_loss_avg = loss_sum / cnt
        train_losses.append(train_loss_avg)
        print(f'Epoch {epoch+1}/{ldm_epochs}, Train Loss: {train_loss_avg:.4f}')

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

            x_flat = x.view(x.shape[0], -1)
            z_0, _, _ = vae.encode(x_flat)
            z_noisy, noise = diffuser.diffusion(z_0, t)
            noise_pred = model(z_noisy, t, labels)
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
        plt.title('LDM Training Loss')
        plt.show()
    else:
        print(f'测试损失: {test_loss_avg:.4f} (未训练，无法显示训练损失曲线)')

# 加载模型（如果不需要训练，也需要加载模型用于推理）
if not train_ldm:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载，路径: {model_path}')
    except FileNotFoundError:
        raise FileNotFoundError(f'未找到模型文件 {model_path}，请先训练模型或指定正确的模型路径')
    except Exception as e:
        raise RuntimeError(f'加载模型时出错: {e}')

# 模型推理 - 生成新图像（使用 LDM + CFG）
if infer:
    model.eval()
    vae.eval()
    print('\n开始生成图像...')
    
    # 生成潜在向量（使用 CFG）
    z_shape = (sample_size, latent_dim)
    labels = None  # None表示随机生成标签
    guidance_scale = config.get('inference', {}).get('guidance_scale', 3.0)
    z, labels = diffuser.sample(
        model, z_shape, labels=labels, guidance_scale=guidance_scale
    )
    
    # 将潜在向量解码为图像
    with torch.no_grad():
        x_hat_flat = vae.decode(z)  # [B, latent_dim] -> [B, C*H*W]
        x_hat = x_hat_flat.view(sample_size, in_channel, img_size, img_size)
    
    # 将张量转换为 PIL 图像
    images = []
    for i in range(sample_size):
        x_img = x_hat[i] * 255
        x_img = x_img.clamp(0, 255)
        x_img = x_img.to(torch.uint8)
        x_img = x_img.cpu()
        to_pil = transforms.ToPILImage()
        images.append(to_pil(x_img))
    
    # 将最终生成的 PIL 图像转换为 numpy 数组用于显示
    images_np = []
    for img in images:
        img_np = np.array(img.convert('L'))  # 转换为灰度图
        images_np.append(img_np)
    
    # 显示最终生成的图像
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle(
        f'Generated Images (LDM+CFG, scale={guidance_scale})',
        fontsize=16, y=0.995
    )

    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(images_np[idx], cmap='gray')
            ax.set_title(f'Label: {labels[idx].item()}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()
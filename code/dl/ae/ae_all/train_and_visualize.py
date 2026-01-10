"""
训练AE、VAE、CVAE模型并可视化2D潜在空间
latent_dim=2用于2D散点图可视化
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models import (AE, VAE, CVAE, CVAE_prior, 
                    ae_loss_function, vae_loss_function, cvae_loss_function, cvae_prior_loss_function)


# 设置参数
input_dim = 784  # 28 * 28
hidden_dim = 128
latent_dim = 2  # 2D可视化
n_classes = 10  # MNIST有10个类别
embed_dim = 5  # embedding维度
batch_size = 64
learning_rate = 0.001
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 获取data文件夹的绝对路径（参考其他文件夹的设置）
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_rel = '../../data'  # 相对于代码文件的路径
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

# 数据集加载
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ========== 可视化函数 ==========
def get_latent_ae(model, data_loader, num_batches=100, device='cpu'):
    """获取AE的潜在空间数据"""
    model.eval()
    z_all = []
    y_all = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.view(-1, input_dim).to(device)
            _, z = model(x)
            z = z.detach().cpu().numpy()
            z_all.append(z)
            y_all.append(y.numpy())
            
            if i >= num_batches:
                break
    
    z_all = np.concatenate(z_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return z_all, y_all


def plot_latent_ae(model, data_loader, num_batches=100, device='cpu'):
    """绘制AE的潜在空间"""
    z_all, y_all = get_latent_ae(model, data_loader, num_batches, device)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar = plt.colorbar(scatter, label='Digit Label', ticks=range(10))
    cbar.set_ticklabels(range(10))
    plt.title('AE Latent Space (2D)')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True, alpha=0.3)
    return z_all, y_all


def get_latent_vae(model, data_loader, num_batches=100, device='cpu'):
    """获取VAE的潜在空间数据（使用mu）"""
    model.eval()
    z_all = []
    y_all = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.view(-1, input_dim).to(device)
            _, mu, _ = model(x)
            z = mu.detach().cpu().numpy()
            z_all.append(z)
            y_all.append(y.numpy())
            
            if i >= num_batches:
                break
    
    z_all = np.concatenate(z_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return z_all, y_all


def plot_latent_vae(model, data_loader, num_batches=100, device='cpu'):
    """绘制VAE的潜在空间（使用mu）"""
    z_all, y_all = get_latent_vae(model, data_loader, num_batches, device)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar = plt.colorbar(scatter, label='Digit Label', ticks=range(10))
    cbar.set_ticklabels(range(10))
    plt.title('VAE Latent Space (2D)')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True, alpha=0.3)
    return z_all, y_all


def get_latent_cvae(model, data_loader, num_batches=100, device='cpu'):
    """获取CVAE的潜在空间数据（使用mu）"""
    model.eval()
    z_all = []
    y_all = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.view(-1, input_dim).to(device)
            y_labels = y.to(device)
            # 直接传递labels，模型内部会转换为embedding
            _, mu, _ = model(x, y_labels)
            z = mu.detach().cpu().numpy()
            z_all.append(z)
            y_all.append(y.numpy())
            
            if i >= num_batches:
                break
    
    z_all = np.concatenate(z_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return z_all, y_all


def plot_latent_cvae(model, data_loader, num_batches=100, device='cpu'):
    """绘制CVAE的潜在空间（使用mu）"""
    z_all, y_all = get_latent_cvae(model, data_loader, num_batches, device)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar = plt.colorbar(scatter, label='Digit Label', ticks=range(10))
    cbar.set_ticklabels(range(10))
    plt.title('CVAE Latent Space (2D)')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True, alpha=0.3)
    return z_all, y_all


# ========== 训练函数 ==========
def train_ae(model, train_loader, epochs, device):
    """训练AE模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    print("训练AE...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, _ in pbar:
            x = x.view(-1, input_dim).to(device)
            
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = ae_loss_function(x_hat, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return model, losses


def train_vae(model, train_loader, epochs, device):
    """训练VAE模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    print("\n训练VAE...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, _ in pbar:
            x = x.view(-1, input_dim).to(device)
            
            optimizer.zero_grad()
            x_hat, mu, sigma = model(x)
            loss = vae_loss_function(x_hat, x, mu, sigma)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return model, losses


def train_cvae(model, train_loader, epochs, device):
    """训练CVAE模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    print("\n训练CVAE...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x = x.view(-1, input_dim).to(device)
            y_labels = y.to(device)
            # 直接传递labels，模型内部会转换为embedding
            
            optimizer.zero_grad()
            x_hat, mu, sigma = model(x, y_labels)
            loss = cvae_loss_function(x_hat, x, mu, sigma)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return model, losses


def get_latent_cvae_prior(model, data_loader, num_batches=100, device='cpu'):
    """获取CVAE_prior的潜在空间数据（使用mu_q）"""
    model.eval()
    z_all = []
    y_all = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.view(-1, input_dim).to(device)
            y_labels = y.to(device)
            # 直接传递labels，模型内部会转换为embedding
            _, mu_q, _, _, _ = model(x, y_labels)
            z = mu_q.detach().cpu().numpy()
            z_all.append(z)
            y_all.append(y.numpy())
            
            if i >= num_batches:
                break
    
    z_all = np.concatenate(z_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return z_all, y_all


def plot_latent_cvae_prior(model, data_loader, num_batches=100, device='cpu'):
    """绘制CVAE_prior的潜在空间（使用mu_q）"""
    z_all, y_all = get_latent_cvae_prior(model, data_loader, num_batches, device)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar = plt.colorbar(scatter, label='Digit Label', ticks=range(10))
    cbar.set_ticklabels(range(10))
    plt.title('CVAE_prior Latent Space (2D)')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True, alpha=0.3)
    return z_all, y_all


def train_cvae_prior(model, train_loader, epochs, device):
    """训练CVAE_prior模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    print("\n训练CVAE_prior...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x = x.view(-1, input_dim).to(device)
            y_labels = y.to(device)
            # 直接传递labels，模型内部会转换为embedding
            
            optimizer.zero_grad()
            x_hat, mu_q, sigma_q, mu_prior, sigma_prior = model(x, y_labels)
            loss = cvae_prior_loss_function(x_hat, x, mu_q, sigma_q, mu_prior, sigma_prior)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return model, losses


# ========== 主程序 ==========
if __name__ == '__main__':
    # 模型文件路径
    ae_model_path = 'ae_model.pth'
    vae_model_path = 'vae_model.pth'
    cvae_model_path = 'cvae_model.pth'
    cvae_prior_model_path = 'cvae_prior_model.pth'
    
    # 创建模型
    ae_model = AE(input_dim, hidden_dim, latent_dim)
    vae_model = VAE(input_dim, hidden_dim, latent_dim)
    cvae_model = CVAE(input_dim, n_classes, embed_dim, hidden_dim, latent_dim)
    cvae_prior_model = CVAE_prior(input_dim, n_classes, embed_dim, hidden_dim, latent_dim)
    
    # 训练或加载AE模型
    if os.path.exists(ae_model_path):
        print(f"加载AE模型: {ae_model_path}")
        ae_model.load_state_dict(torch.load(ae_model_path, weights_only=True))
        ae_model = ae_model.to(device)
        ae_losses = []
    else:
        print("训练AE模型...")
        ae_model, ae_losses = train_ae(ae_model, train_loader, epochs, device)
        torch.save(ae_model.state_dict(), ae_model_path)
        print(f"AE模型已保存: {ae_model_path}")
    
    # 训练或加载VAE模型
    if os.path.exists(vae_model_path):
        print(f"\n加载VAE模型: {vae_model_path}")
        vae_model.load_state_dict(torch.load(vae_model_path, weights_only=True))
        vae_model = vae_model.to(device)
        vae_losses = []
    else:
        print("\n训练VAE模型...")
        vae_model, vae_losses = train_vae(vae_model, train_loader, epochs, device)
        torch.save(vae_model.state_dict(), vae_model_path)
        print(f"VAE模型已保存: {vae_model_path}")
    
    # 训练或加载CVAE模型
    if os.path.exists(cvae_model_path):
        print(f"\n加载CVAE模型: {cvae_model_path}")
        cvae_model.load_state_dict(torch.load(cvae_model_path, weights_only=True))
        cvae_model = cvae_model.to(device)
        cvae_losses = []
    else:
        print("\n训练CVAE模型...")
        cvae_model, cvae_losses = train_cvae(cvae_model, train_loader, epochs, device)
        torch.save(cvae_model.state_dict(), cvae_model_path)
        print(f"CVAE模型已保存: {cvae_model_path}")
    
    # 训练或加载CVAE_prior模型
    if os.path.exists(cvae_prior_model_path):
        print(f"\n加载CVAE_prior模型: {cvae_prior_model_path}")
        cvae_prior_model.load_state_dict(torch.load(cvae_prior_model_path, weights_only=True))
        cvae_prior_model = cvae_prior_model.to(device)
        cvae_prior_losses = []
    else:
        print("\n训练CVAE_prior模型...")
        cvae_prior_model, cvae_prior_losses = train_cvae_prior(cvae_prior_model, train_loader, epochs, device)
        torch.save(cvae_prior_model.state_dict(), cvae_prior_model_path)
        print(f"CVAE_prior模型已保存: {cvae_prior_model_path}")
    
    # 获取潜在空间数据
    print("\n获取潜在空间数据...")
    z_ae, y_ae = get_latent_ae(ae_model, train_loader, num_batches=100, device=device)
    z_vae, y_vae = get_latent_vae(vae_model, train_loader, num_batches=100, device=device)
    z_cvae, y_cvae = get_latent_cvae(cvae_model, train_loader, num_batches=100, device=device)
    z_cvae_prior, y_cvae_prior = get_latent_cvae_prior(cvae_prior_model, train_loader, num_batches=100, device=device)
    
    # 生成四个独立的图
    print("\n生成四个独立的图...")
    plot_latent_ae(ae_model, train_loader, num_batches=100, device=device)
    plt.savefig('ae_latent_space.png', dpi=150, bbox_inches='tight')
    print("AE潜在空间已保存: ae_latent_space.png")
    
    plot_latent_vae(vae_model, train_loader, num_batches=100, device=device)
    plt.savefig('vae_latent_space.png', dpi=150, bbox_inches='tight')
    print("VAE潜在空间已保存: vae_latent_space.png")
    
    plot_latent_cvae(cvae_model, train_loader, num_batches=100, device=device)
    plt.savefig('cvae_latent_space.png', dpi=150, bbox_inches='tight')
    print("CVAE潜在空间已保存: cvae_latent_space.png")
    
    plot_latent_cvae_prior(cvae_prior_model, train_loader, num_batches=100, device=device)
    plt.savefig('cvae_prior_latent_space.png', dpi=150, bbox_inches='tight')
    print("CVAE_prior潜在空间已保存: cvae_prior_latent_space.png")
    
    # 生成四个对比图
    print("\n生成对比图...")
    
    # 对比图1: AE vs VAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter1 = axes[0].scatter(z_ae[:, 0], z_ae[:, 1], c=y_ae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], label='Digit Label', ticks=range(10))
    cbar1.set_ticklabels(range(10))
    axes[0].set_title('AE Latent Space', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('z[0]')
    axes[0].set_ylabel('z[1]')
    axes[0].grid(True, alpha=0.3)
    
    scatter2 = axes[1].scatter(z_vae[:, 0], z_vae[:, 1], c=y_vae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar2 = plt.colorbar(scatter2, ax=axes[1], label='Digit Label', ticks=range(10))
    cbar2.set_ticklabels(range(10))
    axes[1].set_title('VAE Latent Space', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('z[0]')
    axes[1].set_ylabel('z[1]')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_ae_vae.png', dpi=150, bbox_inches='tight')
    print("对比图 AE vs VAE 已保存: comparison_ae_vae.png")
    
    # 对比图2: VAE vs CVAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter1 = axes[0].scatter(z_vae[:, 0], z_vae[:, 1], c=y_vae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], label='Digit Label', ticks=range(10))
    cbar1.set_ticklabels(range(10))
    axes[0].set_title('VAE Latent Space', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('z[0]')
    axes[0].set_ylabel('z[1]')
    axes[0].grid(True, alpha=0.3)
    
    scatter2 = axes[1].scatter(z_cvae[:, 0], z_cvae[:, 1], c=y_cvae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar2 = plt.colorbar(scatter2, ax=axes[1], label='Digit Label', ticks=range(10))
    cbar2.set_ticklabels(range(10))
    axes[1].set_title('CVAE Latent Space', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('z[0]')
    axes[1].set_ylabel('z[1]')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_vae_cvae.png', dpi=150, bbox_inches='tight')
    print("对比图 VAE vs CVAE 已保存: comparison_vae_cvae.png")
    
    # 对比图3: CVAE vs CVAE_prior
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter1 = axes[0].scatter(z_cvae[:, 0], z_cvae[:, 1], c=y_cvae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], label='Digit Label', ticks=range(10))
    cbar1.set_ticklabels(range(10))
    axes[0].set_title('CVAE Latent Space', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('z[0]')
    axes[0].set_ylabel('z[1]')
    axes[0].grid(True, alpha=0.3)
    
    scatter2 = axes[1].scatter(z_cvae_prior[:, 0], z_cvae_prior[:, 1], c=y_cvae_prior, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar2 = plt.colorbar(scatter2, ax=axes[1], label='Digit Label', ticks=range(10))
    cbar2.set_ticklabels(range(10))
    axes[1].set_title('CVAE_prior Latent Space', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('z[0]')
    axes[1].set_ylabel('z[1]')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_cvae_cvae_prior.png', dpi=150, bbox_inches='tight')
    print("对比图 CVAE vs CVAE_prior 已保存: comparison_cvae_cvae_prior.png")
    
    # 对比图4: 四个独立的图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scatter1 = axes[0, 0].scatter(z_ae[:, 0], z_ae[:, 1], c=y_ae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0], label='Digit Label', ticks=range(10))
    cbar1.set_ticklabels(range(10))
    axes[0, 0].set_title('AE Latent Space', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('z[0]')
    axes[0, 0].set_ylabel('z[1]')
    axes[0, 0].grid(True, alpha=0.3)
    
    scatter2 = axes[0, 1].scatter(z_vae[:, 0], z_vae[:, 1], c=y_vae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1], label='Digit Label', ticks=range(10))
    cbar2.set_ticklabels(range(10))
    axes[0, 1].set_title('VAE Latent Space', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('z[0]')
    axes[0, 1].set_ylabel('z[1]')
    axes[0, 1].grid(True, alpha=0.3)
    
    scatter3 = axes[1, 0].scatter(z_cvae[:, 0], z_cvae[:, 1], c=y_cvae, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0], label='Digit Label', ticks=range(10))
    cbar3.set_ticklabels(range(10))
    axes[1, 0].set_title('CVAE Latent Space', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('z[0]')
    axes[1, 0].set_ylabel('z[1]')
    axes[1, 0].grid(True, alpha=0.3)
    
    scatter4 = axes[1, 1].scatter(z_cvae_prior[:, 0], z_cvae_prior[:, 1], c=y_cvae_prior, cmap='viridis', alpha=0.6, s=40, vmin=0, vmax=9)
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 1], label='Digit Label', ticks=range(10))
    cbar4.set_ticklabels(range(10))
    axes[1, 1].set_title('CVAE_prior Latent Space', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('z[0]')
    axes[1, 1].set_ylabel('z[1]')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_all_four.png', dpi=150, bbox_inches='tight')
    print("对比图 四个独立图 已保存: comparison_all_four.png")
    
    # 绘制训练损失曲线（只有在训练时才生成）
    if ae_losses or vae_losses or cvae_losses or cvae_prior_losses:
        plt.figure(figsize=(16, 4))
        num_plots = sum([bool(ae_losses), bool(vae_losses), bool(cvae_losses), bool(cvae_prior_losses)])
        plot_idx = 1
        
        if ae_losses:
            plt.subplot(1, 4, plot_idx)
            plt.plot(ae_losses)
            plt.title('AE Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plot_idx += 1
        
        if vae_losses:
            plt.subplot(1, 4, plot_idx)
            plt.plot(vae_losses)
            plt.title('VAE Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plot_idx += 1
        
        if cvae_losses:
            plt.subplot(1, 4, plot_idx)
            plt.plot(cvae_losses)
            plt.title('CVAE Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plot_idx += 1
        
        if cvae_prior_losses:
            plt.subplot(1, 4, plot_idx)
            plt.plot(cvae_prior_losses)
            plt.title('CVAE_prior Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
        print("\n训练损失曲线已保存: training_losses.png")
    
    plt.show()


"""
测试2: VAE潜在空间插值动画
获取一张label为数字1的图片和一张数字为8的图片，送入VAE编码器得到z1和z8
然后做插值：z = alpha * z1 + (1 - alpha) * z8，alpha从0渐变到1
观察生成图片的渐变效果（动画）
"""
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from models import VAE


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
print("加载VAE模型...")
vae_model = VAE(input_dim, hidden_dim, latent_dim)
vae_model_path = 'vae_model.pth'

if os.path.exists(vae_model_path):
    vae_model.load_state_dict(torch.load(vae_model_path, weights_only=True))
    vae_model = vae_model.to(device)
    vae_model.eval()
    print(f"VAE模型已加载: {vae_model_path}")
else:
    raise FileNotFoundError(f"VAE模型文件不存在: {vae_model_path}")

# 加载数据集
print("\n加载测试数据集...")
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
# 打乱数据集以获取不同的样本
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# 获取数字1和数字8的图片
print("查找数字1和数字8的图片...")
target_labels = [1, 8]
target_images = {}
target_z = {}

with torch.no_grad():
    for x, y in test_loader:
        for label in target_labels:
            if label not in target_images:
                # 找到第一个匹配的标签
                indices = (y == label).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    idx = indices[0].item()
                    img = x[idx:idx+1].view(1, input_dim).to(device)
                    target_images[label] = img.cpu().view(28, 28).numpy()
                    
                    # 编码得到z
                    mu, sigma = vae_model.encoder(img)
                    # 使用mu作为z（也可以使用采样）
                    z = mu
                    target_z[label] = z.cpu().numpy()[0]
                    print(f"找到数字{label}，z = {target_z[label]}")
        
        if len(target_images) == len(target_labels):
            break

if len(target_images) != len(target_labels):
    raise ValueError(f"未能找到所有目标数字，已找到: {list(target_images.keys())}")

z1 = torch.tensor(target_z[1], dtype=torch.float32).to(device)
z8 = torch.tensor(target_z[8], dtype=torch.float32).to(device)

print(f"\nz1 (数字1): {z1.cpu().numpy()}")
print(f"z8 (数字8): {z8.cpu().numpy()}")

# 生成插值序列
num_frames = 60  # 动画帧数
alphas = np.linspace(1, 0, num_frames)  # 从1到0，即从z1渐变到z8

print(f"\n生成{num_frames}帧插值动画...")
interpolated_images = []

with torch.no_grad():
    for alpha in alphas:
        # z = alpha * z1 + (1 - alpha) * z8
        z_interp = alpha * z1 + (1 - alpha) * z8
        # 解码
        img = vae_model.decoder(z_interp.unsqueeze(0))
        img = img.view(28, 28).cpu().numpy()
        interpolated_images.append(img)

# 创建动画
print("创建动画...")
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 上方显示init和final图片
ax_init = fig.add_subplot(gs[0, 0])
ax_final = fig.add_subplot(gs[0, 1])
ax_interp = fig.add_subplot(gs[1, :])

ax_init.imshow(target_images[1], cmap='gray')
ax_init.set_title('Init: Digit 1', fontsize=14, fontweight='bold')
ax_init.axis('off')

ax_final.imshow(target_images[8], cmap='gray')
ax_final.set_title('Final: Digit 8', fontsize=14, fontweight='bold')
ax_final.axis('off')

im = ax_interp.imshow(interpolated_images[0], cmap='gray')
ax_interp.set_title('Interpolated Image (alpha=0.00)', fontsize=14)
ax_interp.axis('off')

title_text = ax_interp.text(0.5, 1.15, '', transform=ax_interp.transAxes, 
                          ha='center', fontsize=14, fontweight='bold')

def animate(frame):
    alpha = alphas[frame]
    im.set_array(interpolated_images[frame])
    title_text.set_text(f'Interpolated Image (alpha={alpha:.2f})\nz = {alpha:.2f} * z1 + {1-alpha:.2f} * z8')
    return [im, title_text]

# 底部添加文字说明（显示当前z值）
info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10)

def update_info(frame):
    alpha = alphas[frame]
    z_interp = alpha * z1 + (1 - alpha) * z8
    info_text.set_text(f'Current z: [{z_interp[0].item():.3f}, {z_interp[1].item():.3f}]')
    return [info_text]

def combined_animate(frame):
    result1 = animate(frame)
    result2 = update_info(frame)
    return result1 + result2

anim = animation.FuncAnimation(fig, combined_animate, frames=num_frames, 
                               interval=100, blit=False, repeat=True)

# 保存动画为GIF
print("保存动画为GIF...")
anim.save('test2_vae_interpolation.gif', writer='pillow', fps=10)
print("动画已保存: test2_vae_interpolation.gif")

# 同时保存一些关键帧的静态图片
print("\n保存关键帧静态图片...")
key_frames = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
key_alphas = [alphas[i] for i in key_frames]

fig2, axes2 = plt.subplots(1, len(key_frames), figsize=(15, 3))
for i, frame_idx in enumerate(key_frames):
    axes2[i].imshow(interpolated_images[frame_idx], cmap='gray')
    axes2[i].set_title(f'alpha={key_alphas[i]:.2f}', fontsize=12)
    axes2[i].axis('off')

plt.tight_layout()
plt.savefig('test2_vae_keyframes.png', dpi=150, bbox_inches='tight')
print("关键帧已保存: test2_vae_keyframes.png")

plt.show()


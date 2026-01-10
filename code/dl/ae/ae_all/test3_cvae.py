"""
测试3: CVAE条件渐变
利用条件0到9渐变，送入CVAE解码器，生成图片0到9渐变的效果
"""
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from models import CVAE


# 设置参数
input_dim = 784  # 28 * 28
hidden_dim = 128
latent_dim = 2
n_classes = 10  # MNIST有10个类别
embed_dim = 5  # embedding维度（需要与训练时一致）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 获取data文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_rel = '../../data'
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

# 加载模型
print("加载CVAE模型...")
cvae_model = CVAE(input_dim, n_classes, embed_dim, hidden_dim, latent_dim)
cvae_model_path = 'cvae_model.pth'

if os.path.exists(cvae_model_path):
    cvae_model.load_state_dict(torch.load(cvae_model_path, weights_only=True))
    cvae_model = cvae_model.to(device)
    cvae_model.eval()
    print(f"CVAE模型已加载: {cvae_model_path}")
else:
    raise FileNotFoundError(f"CVAE模型文件不存在: {cvae_model_path}")

# 从标准正态分布采样latent z（用于所有条件）
print("\n从标准正态分布采样latent z...")
z = torch.randn(1, latent_dim).to(device)
print(f"采样的z: {z.cpu().numpy()}")

# 生成条件0到9的渐变
print("生成条件0到9渐变...")
num_frames = 60  # 动画帧数
labels = np.linspace(0, 9, num_frames)
conditions = []
generated_images = []

with torch.no_grad():
    for label in labels:
        # 取最近的整数作为标签
        label_int = int(round(label))
        conditions.append(label_int)
        
        # 将标签转换为embedding（模型内部处理）
        label_tensor = torch.tensor(label_int, dtype=torch.long).to(device).unsqueeze(0)
        # 获取embedding
        c = cvae_model.condition_embedding(label_tensor)
        
        # 解码：p(x|z, c)
        img = cvae_model.decoder(z, c)
        img = img.view(28, 28).cpu().numpy()
        generated_images.append(img)

# 创建动画
print("创建动画...")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

im = ax.imshow(generated_images[0], cmap='gray')
# ax.set_title(f'CVAE Condition Interpolation (Label: {conditions[0]})', fontsize=14, fontweight='bold')
ax.axis('off')

title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                     ha='center', fontsize=14, fontweight='bold')

info_text = fig.text(0.5, 0.02, f'z = {z.cpu().numpy()[0]}', ha='center', fontsize=12)

def animate(frame):
    label = conditions[frame]
    im.set_array(generated_images[frame])
    title_text.set_text(f'CVAE Condition Interpolation (Label: {label})')
    return [im, title_text]

anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                               interval=100, blit=False, repeat=True)

# 保存动画为GIF
print("保存动画为GIF...")
anim.save('test3_cvae_condition_interpolation.gif', writer='pillow', fps=10)
print("动画已保存: test3_cvae_condition_interpolation.gif")

# 保存一些关键帧的静态图片
print("\n保存关键帧静态图片...")
key_frames = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
key_labels = [conditions[i] for i in key_frames]

fig2, axes2 = plt.subplots(1, len(key_frames), figsize=(15, 3))
for i, frame_idx in enumerate(key_frames):
    axes2[i].imshow(generated_images[frame_idx], cmap='gray')
    axes2[i].set_title(f'Label: {key_labels[i]}', fontsize=12)
    axes2[i].axis('off')

plt.suptitle('CVAE: Condition Interpolation Key Frames\n(Using fixed z from standard normal)', 
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('test3_cvae_keyframes.png', dpi=150, bbox_inches='tight')
print("关键帧已保存: test3_cvae_keyframes.png")

# 生成一个长的横向渐变条
print("\n生成横向渐变条...")
fig2, ax2 = plt.subplots(1, 1, figsize=(20, 2))
gradient_images = np.hstack([img for img in generated_images])
ax2.imshow(gradient_images, cmap='gray', aspect='auto')
ax2.set_title('CVAE Condition Gradient: 0 → 1 → 2 → ... → 8 → 9', fontsize=14, fontweight='bold')
ax2.axis('off')

# 添加标签
for i in range(10):
    x_pos = i * (num_frames - 1) / 9 * 28
    ax2.text(x_pos, -5, str(i), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('test3_cvae_gradient_bar.png', dpi=150, bbox_inches='tight')
print("渐变条已保存: test3_cvae_gradient_bar.png")

# 尝试不同的z值，观察条件的效果
print("\n生成不同z值的条件渐变对比...")
num_z_samples = 3
z_samples = torch.randn(num_z_samples, latent_dim).to(device)
fig3, axes3 = plt.subplots(num_z_samples, 10, figsize=(20, 3 * num_z_samples))

with torch.no_grad():
    for z_idx, z_sample in enumerate(z_samples):
        for label in range(10):
            # 将标签转换为embedding（模型内部处理）
            label_tensor = torch.tensor(label, dtype=torch.long).to(device).unsqueeze(0)
            c = cvae_model.condition_embedding(label_tensor)
            
            img = cvae_model.decoder(z_sample.unsqueeze(0), c)
            img = img.view(28, 28).cpu().numpy()
            axes3[z_idx, label].imshow(img, cmap='gray')
            if z_idx == 0:
                axes3[z_idx, label].set_title(f'Label {label}', fontsize=10)
            if label == 0:
                axes3[z_idx, label].set_ylabel(f'z={z_sample.cpu().numpy()}', fontsize=10)
            axes3[z_idx, label].axis('off')

plt.suptitle('CVAE: Different z values with conditions 0-9', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('test3_cvae_different_z.png', dpi=150, bbox_inches='tight')
print("不同z值对比图已保存: test3_cvae_different_z.png")

plt.show()


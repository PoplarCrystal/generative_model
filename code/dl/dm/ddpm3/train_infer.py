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
from unet import UNet
from ddpm import Diffuser

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
pred_objective = config['diffusion'].get('pred_objective', 'eps')  # 默认 'eps'
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
train = config['flags']['train']
infer = config['flags']['infer']
model_name = config['model_file']['model_name']
# 如果 model_name 为空，根据 pred_objective 自动生成
if not model_name or model_name.strip() == '':
    model_name = f'ddpm_{pred_objective}_model.pth'
    print(f'模型文件名未指定，根据 pred_objective={pred_objective} 自动生成: {model_name}')
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
model = UNet(in_ch=in_channel, time_embed_dim=time_embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'使用学习率: {learning_rate}')

# 扩散过程管理器
diffuser = Diffuser(
    num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, device=device, pred_objective=pred_objective
)
print(f'使用预测目标: {pred_objective}')

train_losses = []


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
    print('\n开始训练...')
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.diffusion(x, t)
            model_output = model(x_noisy, t)
            
            # 根据预测目标计算损失
            if pred_objective == 'eps':
                target = noise
            elif pred_objective == 'x0':
                target = x
            elif pred_objective == 'score':
                # score = -eps / sqrt(1 - alpha_bar)
                t_idx = t - 1
                alpha_bar = diffuser.alpha_bars[t_idx].view(len(t), 1, 1, 1)
                target = -noise / torch.sqrt(1 - alpha_bar)
            else:
                raise ValueError(f"Unknown pred_objective: {pred_objective}")
            
            loss = F.mse_loss(model_output, target)

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
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.diffusion(x, t)
            model_output = model(x_noisy, t)
            
            # 根据预测目标计算损失
            if pred_objective == 'eps':
                target = noise
            elif pred_objective == 'x0':
                target = x
            elif pred_objective == 'score':
                # score = -eps / sqrt(1 - alpha_bar)
                t_idx = t - 1
                alpha_bar = diffuser.alpha_bars[t_idx].view(len(t), 1, 1, 1)
                target = -noise / torch.sqrt(1 - alpha_bar)
            else:
                raise ValueError(f"Unknown pred_objective: {pred_objective}")
            
            loss = F.mse_loss(model_output, target)

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
        plt.title('DDPM Training Loss')
        plt.show()
    else:
        print(f'测试损失: {test_loss_avg:.4f} (未训练，无法显示训练损失曲线)')

# 加载模型（如果不需要训练，也需要加载模型用于推理）
if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载，路径: {model_path}')
    except FileNotFoundError:
        raise FileNotFoundError(
            f'未找到模型文件 {model_path}，请先训练模型或指定正确的模型路径'
        )
    except Exception as e:
        raise RuntimeError(f'加载模型时出错: {e}')

# 模型推理 - 生成新图像
if infer:
    model.eval()
    print('\n开始生成图像...')
    
    # 生成图像
    x_shape = (sample_size, in_channel, img_size, img_size)
    images = diffuser.sample(model, x_shape)
    
    # 将最终生成的 PIL 图像转换为 numpy 数组用于显示
    images_np = []
    for img in images:
        img_np = np.array(img.convert('L'))  # 转换为灰度图
        images_np.append(img_np)
    
    # 显示最终生成的图像
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle('Generated Images (DDPM)', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(images_np[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()
    
    # 显示去噪过程（16个中间步骤）
    print('\n显示去噪过程...')
    num = num_timesteps / 16  # 每步的步长
    
    # 使用相同的噪声起点，确保可以看到渐进过程
    torch.manual_seed(42)  # 固定随机种子
    noise = torch.randn(x_shape, device=device)
    
    # 在一次完整的去噪过程中，在 num, 2*num, ..., 16*num 步时分别保存中间图像（去噪过程）
    _, intermediate_images, timestep_labels = diffuser.sample_with_intermediate_steps(
        model, x_shape=x_shape, step_interval=num, noise=noise
    )
    
    # 将 PIL 图像转换为 numpy 数组用于显示
    intermediate_images_np = []
    for img in intermediate_images:
        img_np = np.array(img.convert('L'))  # 转换为灰度图
        intermediate_images_np.append(img_np)
    
    # 显示去噪过程
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Denoising Process (from noise to image)', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(intermediate_images_np):
            ax.imshow(intermediate_images_np[idx], cmap='gray')
            ax.set_title(timestep_labels[idx], fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.1)
    plt.show()


import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import numpy as np
from ar_transformer import AR_Transformer, loss_function

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config_transformer.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
vocab_size = config['model']['vocab_size']
d_model = config['model']['d_model']
nhead = config['model']['nhead']
num_layers = config['model']['num_layers']
dim_feedforward = config['model']['dim_feedforward']
dropout = config['model']['dropout']
max_len = config['model']['max_len']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
train = config['flags']['train']
infer = config['flags']['infer']
model_name = config['model_file']['model_name']
seq_len = config['model']['seq_len']  # MNIST: 28*28=784

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
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=8,  # 增加多进程数量以加速数据加载
    pin_memory=True,  # 加速GPU传输
    persistent_workers=True  # 保持worker进程，避免重复创建
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# 模型构建
model = AR_Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    max_len=max_len
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []


def preprocess_images(images):
    """将图像转换为像素序列
    
    Args:
        images: (batch_size, 1, 28, 28) 图像张量，值在[0, 1]范围
    Returns:
        pixel_seq: (batch_size, 784) 像素序列，值在[0, 255]范围（整数）
    """
    # 展平图像
    images_flat = images.reshape(images.size(0), -1)  # (batch_size, 784)
    # 转换为0-255范围并量化
    pixel_seq = (images_flat * 255).long().clamp(0, 255)
    return pixel_seq


def prepare_sequences(pixel_seq):
    """准备训练序列：输入是前n-1个像素，目标是后n-1个像素（shift by 1）
    
    Args:
        pixel_seq: (batch_size, seq_len) 像素序列
    Returns:
        inputs: (batch_size, seq_len-1) 输入序列
        targets: (batch_size, seq_len-1) 目标序列
    """
    inputs = pixel_seq[:, :-1]  # 前n-1个像素作为输入
    targets = pixel_seq[:, 1:]  # 后n-1个像素作为目标（shift by 1）
    return inputs, targets


if train:
    # 模型训练
    print('\n开始训练 AR-Transformer 模型...')
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            # 预处理图像（确保在GPU上）
            images = images.to(device, non_blocking=True)
            pixel_seq = preprocess_images(images)  # (batch_size, 784)
            
            # 准备输入和目标
            inputs, targets = prepare_sequences(pixel_seq)  # (batch_size, 783)
            
            # 前向传播（训练时使用因果掩码）
            logits = model(inputs, use_causal_mask=True)  # (batch_size, 783, vocab_size)
            loss = loss_function(logits, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            pixel_seq = preprocess_images(images)
            inputs, targets = prepare_sequences(pixel_seq)
            logits = model(inputs, use_causal_mask=True)
            loss = loss_function(logits, targets)
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
        plt.title('AR-Transformer Training Loss')
        plt.show()

# 加载模型（如果不需要训练，也需要加载模型用于推理）
if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载，路径: {model_path}')
    except FileNotFoundError:
        print(f'警告: 未找到模型文件 {model_path}，使用随机初始化的模型')
    except Exception as e:
        print(f'加载模型时出错: {e}')

# 模型推理 - 生成新图像
if infer:
    model.eval()
    print('\n开始生成图像...')
    print('注意: AR模型是自回归生成，逐像素生成，速度较慢...')
    
    with torch.no_grad():
        # 生成图像
        generated_sequences = model.generate(
            batch_size=sample_size,
            seq_len=seq_len,
            device=device,
            temperature=1.0
        )  # (sample_size, 784)
        
        # 转换为图像格式：将像素值从[0, 255]转换回[0, 1]
        generated_images = generated_sequences.float() / 255.0
        generated_images = generated_images.view(sample_size, 1, 28, 28).cpu()
    
    # 显示生成的图像
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle('Generated Images (AR-Transformer)', fontsize=16, y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(generated_images[idx].squeeze().numpy(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
    plt.show()

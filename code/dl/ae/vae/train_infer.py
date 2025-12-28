import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from vae import VAE, VAE_CNN, loss_function

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
model_type = config['model']['model_type']  # 'vae' 或 'vae_cnn'
input_dim = config['model']['input_dim']
output_dim = config['model']['output_dim']
hidden_dim = config['model']['hidden_dim']
latent_dim = config['model']['latent_dim']
input_channel = config['model']['input_channel']
output_channel = config['model']['output_channel']
hidden_channel = config['model']['hidden_channel']
latent_channel = config['model']['latent_channel']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
learning_rate_cnn = config['training'].get('learning_rate_cnn', 0.001)  # CNN 默认学习率
batch_size = config['training']['batch_size']
train = config['flags']['train']
infer1 = config['flags']['infer1']
infer2 = config['flags']['infer2']
model_name = config['model_file']['model_name']
# 如果 model_name 为空，根据 model_type 自动生成
if not model_name or model_name.strip() == '':
    model_name = f'{model_type}_model.pth'
    print(f'模型文件名未指定，根据 model_type 自动生成: {model_name}')
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型构建 - 根据 model_type 选择模型
if model_type == 'vae':
    model = VAE(input_dim, output_dim, hidden_dim, latent_dim).to(device)
    use_cnn = False
elif model_type == 'vae_cnn':
    model = VAE_CNN(input_channel, output_channel, hidden_channel, latent_channel).to(device)
    use_cnn = True
else:
    raise ValueError(f"不支持的模型类型: {model_type}，请选择 'vae' 或 'vae_cnn'")

print(f'使用模型类型: {model_type}')
# CNN 使用不同的学习率
if use_cnn:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_cnn)
    print(f'使用 CNN 学习率: {learning_rate_cnn}')
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f'使用 VAE 学习率: {learning_rate}')
train_losses = []

if train:
    # 模型训练
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, label in pbar:
            if use_cnn:
                x = x.to(device)  # (batch_size, 1, 28, 28)
            else:
                x = x.view(-1, input_dim).to(device)  # (batch_size, 784)
            x_hat, mu, sigma = model(x)
            if use_cnn:
                loss = loss_function(x_hat, x, mu, sigma)
            else:
                loss = loss_function(x_hat, x, mu, sigma)
            optimizer.zero_grad()
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
        for x, label in pbar:
            if use_cnn:
                x = x.to(device)  # (batch_size, 1, 28, 28)
            else:
                x = x.view(-1, input_dim).to(device)  # (batch_size, 784)
            x_hat, mu, sigma = model(x)
            loss = loss_function(x_hat, x, mu, sigma)
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
        plt.title(f'{model_type.upper()} Training Loss')
        plt.show()
    else:
        print(f'测试损失: {test_loss_avg:.4f} (未训练，无法显示训练损失曲线)')

# 加载模型（如果不需要训练，也需要加载模型用于推理）
if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载，路径: {model_path}')
    except FileNotFoundError:
        print(f'警告: 未找到模型文件 {model_path}，使用随机初始化的模型')
    except Exception as e:
        print(f'加载模型时出错: {e}')

# 模型推理1 - 从测试集采样图像，查看重构效果
if infer1:
    model.eval()
    with torch.no_grad():
        # 从测试集中收集足够的样本（可能需要多个批次）
        x_test_list = []
        labels_test_list = []
        for x, labels in test_loader:
            x_test_list.append(x)
            labels_test_list.append(labels)
            if len(x_test_list) * batch_size >= sample_size:
                break
        
        # 拼接并截取所需数量的样本
        x_test = torch.cat(x_test_list, dim=0)[:sample_size]
        labels_test = torch.cat(labels_test_list, dim=0)[:sample_size]
        
        if use_cnn:
            x_test = x_test.to(device)  # (sample_size, 1, 28, 28)
        else:
            x_test = x_test.view(-1, input_dim).to(device)  # (sample_size, 784)
        
        # 通过 VAE 进行编码-解码重构
        x_reconstructed, mu, sigma = model(x_test)
        
        # 转换为图像格式用于显示
        if use_cnn:
            original_images = x_test.cpu()
            reconstructed_images = x_reconstructed.cpu()
            # 如果尺寸不匹配，使用插值调整到 28x28
            if reconstructed_images.shape[2] != 28 or reconstructed_images.shape[3] != 28:
                reconstructed_images = F.interpolate(reconstructed_images, size=(28, 28), mode='bilinear', align_corners=False)
        else:
            original_images = x_test.view(sample_size, 1, 28, 28).cpu()
            reconstructed_images = x_reconstructed.view(sample_size, 1, 28, 28).cpu()
        labels_cpu = labels_test.numpy()

    # 显示原始图像和重构图像的对比（左右排布）
    fig, axes = plt.subplots(nrow, nrow * 2, figsize=(16, 8))
    fig.suptitle('Original (Left) vs Reconstructed (Right) Images', fontsize=16, y=0.995)

    for idx in range(sample_size):
        row = idx // nrow
        col_base = (idx % nrow) * 2
        
        # 左：原始图像
        axes[row, col_base].imshow(original_images[idx].squeeze().numpy(), cmap='gray')
        axes[row, col_base].set_title(f'Label: {labels_cpu[idx]}', fontsize=10)
        axes[row, col_base].axis('off')
        
        # 右：重构图像
        axes[row, col_base + 1].imshow(reconstructed_images[idx].squeeze().numpy(), cmap='gray')
        axes[row, col_base + 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.99, hspace=0.2, wspace=0.1)
    plt.show()

# 模型推理2 - 生成新图像（从潜在空间随机采样）
if infer2:
    model.eval()
    with torch.no_grad():
        if use_cnn:
            # 卷积 VAE 的潜在空间是 4D: (batch, latent_channel, 1, 1)
            # 28x28 -> 14x14 -> 7x7 -> 1x1 (经过两次 stride=2 的下采样 + 7x7 conv)
            z = torch.randn(sample_size, latent_channel, 1, 1).to(device)
            x = model.decoder(z)
            # 新架构应该输出精确的 28x28，但为了保险还是检查一下
            if x.shape[2] != 28 or x.shape[3] != 28:
                x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
            generated_images = x.cpu()
        else:
            z = torch.randn(sample_size, latent_dim).to(device)
            x = model.decoder(z)
            generated_images = x.view(sample_size, 1, 28, 28).cpu()

    grid_img = torchvision.utils.make_grid(generated_images, nrow=nrow, padding=2, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title('Generated Images')
    plt.show()


import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from vae_cg import (
    VAE_CG, VAE_CFG,
    ConditionEncoder,
    loss_function_cg, loss_function_cfg
)

# 加载配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
model_type = config['model']['model_type']  # 'vae_cg' 或 'vae_cfg'
input_dim = config['model']['input_dim']
output_dim = config['model']['output_dim']
n_classes = config['model']['n_classes']
condition_encoding_type = config['model'].get('condition_encoding_type', 'onehot')
embed_dim = config['model'].get('embed_dim', None) if condition_encoding_type == 'embedding' else None
hidden_dim = config['model']['hidden_dim']
latent_dim = config['model']['latent_dim']
unconditional_prob = config['model'].get('unconditional_prob', 0.1)  # CFG训练时的无条件概率
classification_weight = config['model'].get('classification_weight', 0.1)  # CG的分类损失权重
guidance_scale = config['inference'].get('guidance_scale', 1.5)  # 引导强度

epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
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

# 创建条件编码器
condition_encoder = ConditionEncoder(
    n_classes=n_classes,
    condition_encoding_type=condition_encoding_type,
    embed_dim=embed_dim
).to(device)
condition_dim = condition_encoder.condition_dim
print(f'条件编码方式: {condition_encoding_type}, 条件维度: {condition_dim}')

# 模型构建
if model_type == 'vae_cg':
    model = VAE_CG(input_dim, output_dim, n_classes, hidden_dim, latent_dim,
                   condition_encoder=condition_encoder).to(device)
elif model_type == 'vae_cfg':
    model = VAE_CFG(input_dim, output_dim, n_classes, hidden_dim, latent_dim,
                    condition_encoder=condition_encoder,
                    unconditional_prob=unconditional_prob).to(device)
else:
    raise ValueError(f"不支持的模型类型: {model_type}，请选择 'vae_cg' 或 'vae_cfg'")

print(f'使用模型类型: {model_type}')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []

if train:
    # 模型训练
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, label in pbar:
            x = x.view(-1, input_dim).to(device)
            labels = label.to(device)
            
            if model_type == 'vae_cg':
                # VAE_CG: 需要分类器输出
                x_hat, mu, sigma, class_logits = model(x, labels)
                loss = loss_function_cg(x_hat, x, mu, sigma, class_logits, labels, classification_weight)
            elif model_type == 'vae_cfg':
                # VAE_CFG: 随机丢弃条件（内部处理）
                x_hat, mu, sigma = model(x, labels)
                loss = loss_function_cfg(x_hat, x, mu, sigma)
            
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
            x = x.view(-1, input_dim).to(device)
            labels = label.to(device)
            
            if model_type == 'vae_cg':
                x_hat, mu, sigma, class_logits = model(x, labels)
                loss = loss_function_cg(x_hat, x, mu, sigma, class_logits, labels, classification_weight)
            elif model_type == 'vae_cfg':
                x_hat, mu, sigma = model(x, labels)
                loss = loss_function_cfg(x_hat, x, mu, sigma)
            
            test_loss_sum += loss.item()
            test_cnt += 1
            pbar.set_postfix({'test_loss': f'{test_loss_sum/test_cnt:.4f}'})

    test_loss_avg = test_loss_sum / test_cnt
    print(f'Test Loss: {test_loss_avg:.4f}\n')

    # 训练损失曲线
    if len(train_losses) > 0:
        epochs_list = list(range(1, len(train_losses) + 1))
        plt.plot(epochs_list, train_losses, marker='o', linestyle='-', label='Train Loss')
        plt.axhline(y=test_loss_avg, color='r', linestyle='--', label=f'Test Loss: {test_loss_avg:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_type.upper()} Training Loss')
        plt.show()

# 加载模型
if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载，路径: {model_path}')
    except FileNotFoundError:
        print(f'警告: 未找到模型文件 {model_path}，使用随机初始化的模型')
    except Exception as e:
        print(f'加载模型时出错: {e}')

# 推理1 - 重构测试
if infer1:
    model.eval()
    with torch.no_grad():
        x_test_list = []
        labels_test_list = []
        for x, labels in test_loader:
            x_test_list.append(x)
            labels_test_list.append(labels)
            if len(x_test_list) * batch_size >= sample_size:
                break
        
        x_test = torch.cat(x_test_list, dim=0)[:sample_size].view(-1, input_dim).to(device)
        labels_test = torch.cat(labels_test_list, dim=0)[:sample_size].to(device)
        
        if model_type == 'vae_cg':
            x_reconstructed, mu, sigma, class_logits = model(x_test, labels_test)
        elif model_type == 'vae_cfg':
            x_reconstructed, mu, sigma = model(x_test, labels_test)
        
        original_images = x_test.view(sample_size, 1, 28, 28).cpu()
        reconstructed_images = x_reconstructed.view(sample_size, 1, 28, 28).cpu()
        labels_cpu = labels_test.cpu().numpy()

    fig, axes = plt.subplots(nrow, nrow * 2, figsize=(16, 8))
    fig.suptitle('Original (Left) vs Reconstructed (Right) Images', fontsize=16, y=0.995)

    for idx in range(sample_size):
        row = idx // nrow
        col_base = (idx % nrow) * 2
        
        axes[row, col_base].imshow(original_images[idx].squeeze().numpy(), cmap='gray')
        axes[row, col_base].set_title(f'Label: {labels_cpu[idx]}', fontsize=10)
        axes[row, col_base].axis('off')
        
        axes[row, col_base + 1].imshow(reconstructed_images[idx].squeeze().numpy(), cmap='gray')
        axes[row, col_base + 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.99, hspace=0.2, wspace=0.1)
    plt.show()

# 推理2 - 引导生成
if infer2:
    # 随机采样条件
    conditions = torch.randint(0, n_classes, (sample_size,)).to(device)
    
    if model_type == 'vae_cg':
        # Classifier Guidance生成（需要在enable_grad上下文中）
        # 注意：sample_with_classifier_guidance内部已经处理了梯度计算
        model.eval()  # 模型设为eval，但分类器会在函数内部设为train
        z = model.sample_with_classifier_guidance(conditions, guidance_scale=guidance_scale)
        # 生成图像时不需要梯度
        with torch.no_grad():
            c = condition_encoder(conditions)
            x = model.decoder(z, c)
            generated_images = x.view(sample_size, 1, 28, 28).cpu()
        
    elif model_type == 'vae_cfg':
        # Classifier-Free Guidance生成
        model.eval()
        with torch.no_grad():
            x_guided, x_cond, x_uncond = model.sample_with_cfg(conditions, guidance_scale=guidance_scale, num_samples=sample_size)
            generated_images = x_guided.view(sample_size, 1, 28, 28).cpu()
    
    conditions_cpu = conditions.cpu().numpy()

    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
    fig.suptitle(f'Generated Images with {model_type.upper()} (guidance_scale={guidance_scale})', fontsize=16, y=0.995)

    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(generated_images[idx].squeeze().numpy(), cmap='gray')
            ax.text(0.5, -0.1, str(int(conditions_cpu[idx])), 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.3, wspace=0.1)
    plt.show()


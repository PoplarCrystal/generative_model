import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        z = self.linear2(h)  # 直接输出潜在向量，不需要概率分布
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        x_hat = F.sigmoid(self.linear2(h))
        return x_hat


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)  # 编码：x -> z
        x_hat = self.decoder(z)  # 解码：z -> x_hat
        return x_hat


def loss_function(x_hat, x):
    # AE 只有重构损失，没有 KL 散度
    return F.mse_loss(x_hat, x, reduction='mean')


# 第一步：超参数设置
input_dim = 28 * 28
hidden_dim = 128
latent_dim = 16
epochs = 30
learning_rate = 3e-4
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 第二步：数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 第三步：模型构建
model = AE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
train_model = False
model_name = 'ae_model.pth'

if train_model:
    # 第四步：模型训练
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, label in pbar:
            x = x.view(-1, input_dim).to(device)
            x_hat = model(x)
            loss = loss_function(x_hat, x)
            
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
    torch.save(model.state_dict(), model_name)
    print(f'模型已保存，文件名: {model_name}')

# 加载模型
try:
    model.load_state_dict(torch.load(model_name, weights_only=True))
    print(f'模型已加载，文件名: {model_name}')
except FileNotFoundError:
    print(f'警告: 未找到模型文件 {model_name}，使用随机初始化的模型')
except Exception as e:
    print(f'加载模型时出错: {e}')

# 第五步：模型测试
print('\n开始测试...')
model.eval()
test_loss_sum = 0.0
test_cnt = 0
with torch.no_grad():
    pbar = tqdm(test_loader, desc='Testing')
    for x, label in pbar:
        x = x.view(-1, input_dim).to(device)
        x_hat = model(x)
        loss = loss_function(x_hat, x)
        test_loss_sum += loss.item()
        test_cnt += 1
        pbar.set_postfix({'test_loss': f'{test_loss_sum/test_cnt:.4f}'})

test_loss_avg = test_loss_sum / test_cnt
print(f'Test Loss: {test_loss_avg:.4f}\n')

# 第六步：模型可视化
if len(train_losses) > 0:
    epochs_list = list(range(1, len(train_losses) + 1))
    plt.plot(epochs_list, train_losses, marker='o', linestyle='-', label='Train Loss')
    plt.axhline(y=test_loss_avg, color='r', linestyle='--', label=f'Test Loss: {test_loss_avg:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('AE Training Loss')
    plt.show()
else:
    print(f'测试损失: {test_loss_avg:.4f} (未训练，无法显示训练损失曲线)')

# 第七步：模型推理 - 从测试集采样图像，查看重构效果
model.eval()
with torch.no_grad():
    sample_size = 16
    nrow = 4
    
    # 从测试集中收集足够的样本（可能需要多个批次）
    x_test_list = []
    labels_test_list = []
    for x, labels in test_loader:
        x_test_list.append(x)
        labels_test_list.append(labels)
        if len(x_test_list) * batch_size >= sample_size:
            break
    
    # 拼接并截取所需数量的样本
    x_test = torch.cat(x_test_list, dim=0)[:sample_size].view(-1, input_dim).to(device)
    labels_test = torch.cat(labels_test_list, dim=0)[:sample_size]
    
    # 通过 AE 进行编码-解码重构
    x_reconstructed = model(x_test)
    
    # 转换为图像格式用于显示
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

# 第八步：生成新图像（从潜在空间随机采样）
# model.eval()
# with torch.no_grad():
#     sample_size = 16
#     nrow = 4
#     # 从标准正态分布采样潜在向量
#     z = torch.randn(sample_size, latent_dim).to(device)
#     x = model.decoder(z)
#     generated_images = x.view(sample_size, 1, 28, 28).cpu()

# # 显示生成的图像
# fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
# fig.suptitle('Generated Images (AE)', fontsize=16, y=0.995)

# for idx, ax in enumerate(axes.flat):
#     if idx < sample_size:
#         ax.imshow(generated_images[idx].squeeze().numpy(), cmap='gray')
#     ax.axis('off')

# plt.tight_layout()
# plt.subplots_adjust(top=0.98, hspace=0.1, wspace=0.1)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

# 原始标签：lbls = tensor([3, 7, 1, ...])
# 转换为 one-hot：condition = tensor([
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 数字3
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 数字7
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 数字1
#     ...
# ])
def onehot(x, max_dim):
    # 方法1：使用 F.one_hot（最简单，推荐）
    return F.one_hot(x, num_classes=max_dim).float()

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.linear(x))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat


class CVAE(nn.Module):
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()
        # self.encoder = Encoder(input_dim + condition_dim, hidden_dim, latent_dim)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + condition_dim, hidden_dim, output_dim)
    
    def forward(self, x, c):
        # x = torch.cat((x, c), dim=1)
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
        z = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

    def _reparameterize(self, mu, sigma): # 重参数化
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z


def loss_function(x_hat, x, mu, sigma):
    batch_size = x.shape[0]
    L1 = F.mse_loss(x_hat, x, reduction='sum')
    L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
    return (L1 + L2) / batch_size


# 第一步： 超参数设置
input_dim = 28*28  # x dimension
output_dim = 28*28  # x dimension
condition_dim = 10  # condition dimension
hidden_dim = 128  # neurons in hidden layers
latent_dim = 16  # z dimension
epochs = 30
learning_rate = 3e-4
batch_size = 32
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备
print(f'设备: {device}')

# 第二步： 数据集加载
transform = transforms.Compose([
                transforms.ToTensor(),
            ])
# MNIST 数据集已经预分好了训练集和测试集，不需要手动分配比例
# train=True: 60,000 张训练图片
# train=False: 10,000 张测试图片
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   

# 第三步： 模型构建
model = CVAE(input_dim, output_dim, condition_dim, hidden_dim, latent_dim).to(device)   
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
train_model = True
model_name = 'cvae1_model.pth'

if train_model:
    # 第四步： 模型训练
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, label in pbar:
            x = x.view(-1, input_dim).to(device) #batch_size,input_dim(28*28)
            c = onehot(label, condition_dim).to(device) #batch_size,condition_dim
            x_hat, mu, sigma = model(x, c) # x_hat: (batch_size, input_dim), mu: (batch_size, latent_dim), sigma: (batch_size, latent_dim)
            loss = loss_function(x_hat, x, mu, sigma)
            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
            
            # 更新进度条显示当前平均损失
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
# 第五步： 模型测试
print('\n开始测试...')
model.eval()
test_loss_sum = 0.0
test_cnt = 0
with torch.no_grad():
    pbar = tqdm(test_loader, desc='Testing')
    for x, label in pbar:
        x = x.view(-1, input_dim).to(device)
        c = onehot(label, condition_dim).to(device) #batch_size,condition_dim
        x_hat, mu, sigma = model(x, c)
        loss = loss_function(x_hat, x, mu, sigma)
        test_loss_sum += loss.item()
        test_cnt += 1
        pbar.set_postfix({'test_loss': f'{test_loss_sum/test_cnt:.4f}'})

test_loss_avg = test_loss_sum / test_cnt
print(f'Test Loss: {test_loss_avg:.4f}\n')

# 第六步： 模型可视化
if train_model:
    epochs_list = list(range(1, len(train_losses) + 1))
    plt.plot(epochs_list, train_losses, marker='o', linestyle='-', label='Train Loss')
    plt.axhline(y=test_loss_avg, color='r', linestyle='--', label=f'Test Loss: {test_loss_avg:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Loss')
    plt.show()
else:
    print(f'测试损失: {test_loss_avg:.4f} (未训练，无法显示训练损失曲线)')

# 第七步： 生成新图像（带条件标签）
model.eval()
with torch.no_grad():
    sample_size = 16
    nrow = 4  # 每行显示4个图像
    # 1. 随机采样条件（0-9的数字）
    conditions = torch.randint(0, 10, (sample_size,)).to(device)  # 随机生成16个0-9的数字
    c = onehot(conditions, condition_dim).to(device)  # 转换为one-hot编码
    # 2. 从标准正态分布采样潜在向量
    z = torch.randn(sample_size, latent_dim).to(device)
    # 3. 将潜在向量和条件拼接后输入decoder
    z_with_condition = torch.cat((z, c), dim=1)  # (sample_size, latent_dim + condition_dim)
    x = model.decoder(z_with_condition)
    generated_images = x.view(sample_size, 1, 28, 28).cpu()
    conditions_cpu = conditions.cpu().numpy()  # 转换为numpy用于显示

# 4. 显示图像和标签（4×4网格，每个格子包含图像和标签）
fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
fig.suptitle('Generated Images with Condition Labels', fontsize=16, y=0.995)

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
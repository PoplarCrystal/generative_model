import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_channel, hidden_channel, latent_channel):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, hidden_channel, kernel_size=13)
        self.conv_mu = nn.Conv2d(hidden_channel, latent_channel, kernel_size=13)  # 输出通道数应该是 latent_channel
        self.conv_logvar = nn.Conv2d(hidden_channel, latent_channel, kernel_size=13)

    def forward(self, x):
        h = F.relu(self.conv(x))
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_channel, hidden_channel, output_channel):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_channel, hidden_channel, kernel_size=13)
        self.conv2 = nn.ConvTranspose2d(hidden_channel, output_channel, kernel_size=13)

    def forward(self, z):
        h = F.relu(self.conv1(z))
        x_hat = torch.sigmoid(self.conv2(h))  # 添加 sigmoid 确保输出在 [0,1] 范围
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channel, latent_channel):
        super().__init__()
        self.encoder = Encoder(input_channel, hidden_channel, latent_channel)
        self.decoder = Decoder(latent_channel, hidden_channel, output_channel)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self._reparameterize(mu, sigma)
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
input_channel = 1  # x dimension
output_channel = 1  # x dimension
hidden_channel = 4  # neurons in hidden layers
latent_channel = 2  # z dimension
epochs = 40
learning_rate = 1e-3
batch_size = 32
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备
print(f'设备: {device}')

# 第二步： 数据集加载
# 卷积网络需要保持图像的空间结构，不要展平
transform = transforms.Compose([
                transforms.ToTensor(),
                # 不展平，保持 (1, 28, 28) 格式
            ])
# MNIST 数据集已经预分好了训练集和测试集，不需要手动分配比例
# train=True: 60,000 张训练图片
# train=False: 10,000 张测试图片
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   

# 第三步： 模型构建
model = VAE(input_channel, output_channel, hidden_channel, latent_channel).to(device)   
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
train_model = True
model_name = 'vae_cnn_model.pth'

if train_model:
    # 第四步： 模型训练
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, label in pbar:
            x = x.view(-1, input_channel, 28, 28).to(device)
            x_hat, mu, sigma = model(x)  # x_hat: (batch_size, 1, 28, 28), mu/sigma: (batch_size, latent_channel, 4, 4)
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
        x = x.view(-1, input_channel, 28, 28).to(device)
        x_hat, mu, sigma = model(x)
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

# 第七步： 生成新图像
model.eval()
with torch.no_grad():
    sample_size = 16
    # 卷积 VAE 的潜在空间是 4D: (batch, latent_channel, 4, 4)
    # 28x28 图像经过两个 kernel_size=13 的卷积后变成 4x4
    z = torch.randn(sample_size, latent_channel, 4, 4).to(device)
    x = model.decoder(z)  # 输出应该是 (sample_size, 1, H, W)
    # decoder 输出可能不是精确的 28x28，需要调整或裁剪
    if x.shape[2] != 28 or x.shape[3] != 28:
        # 如果尺寸不匹配，使用插值调整到 28x28
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
    generated_images = x.cpu()

grid_img = torchvision.utils.make_grid(generated_images, nrow=4, padding=2, normalize=True)
plt.imshow(grid_img.permute(1, 2, 0).numpy())
plt.axis('off')
plt.title('Generated Images')
plt.show()
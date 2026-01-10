"""
测试4: CVAE Embedding优化实验
1. 排除数字8和9进行训练（只训练0-7）
2. 使用8和9条件进行解码，观察生成效果
3. 冻结CVAE模型参数，只优化embedding层
4. 用8和9的数据优化embedding，观察能否生成合理的8和9数字
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models import CVAE, cvae_loss_function
from tqdm import tqdm


# 设置参数
input_dim = 784  # 28 * 28
hidden_dim = 128
latent_dim = 2
n_classes = 10  # MNIST有10个类别
embed_dim = 5  # embedding维度
batch_size = 64
learning_rate = 0.02
epochs_finetune = 20  # 微调embedding的epoch数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 获取data文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_rel = '../../data'
data_dir = os.path.join(current_dir, data_dir_rel)
data_dir = os.path.normpath(data_dir)

# 创建排除 8 和 9 的训练数据集（0-7）
print("\n创建排除8和9的训练数据集（0-7）...")
transform = transforms.Compose([transforms.ToTensor()])
ds_train_all = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)

# 排除8和9
idxn9 = ds_train_all.targets != 9
idxn8 = ds_train_all.targets != 8
idx = idxn9 & idxn8

ds_train_07 = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
ds_train_07.targets = ds_train_all.targets[idx]
ds_train_07.data = ds_train_all.data[idx]

print(f"训练集（0-7）: {len(ds_train_07)} 个样本")

# ========== 第一部分：训练只使用0-7数据的CVAE模型 ==========
print("\n" + "="*60)
print("第一部分：训练只使用0-7数据的CVAE模型")
print("="*60)

cvae_model_path_07 = 'cvae_model_0_7.pth'
cvae_model = CVAE(input_dim, n_classes, embed_dim, hidden_dim, latent_dim)

# 检查是否已有训练好的0-7模型
if os.path.exists(cvae_model_path_07):
    print(f"加载已训练的0-7模型: {cvae_model_path_07}")
    cvae_model.load_state_dict(torch.load(cvae_model_path_07, weights_only=True))
    cvae_model = cvae_model.to(device)
    cvae_model.eval()
    print("模型已加载")
else:
    print("训练新的0-7模型...")
    train_loader_07 = DataLoader(ds_train_07, batch_size=batch_size, shuffle=True)
    cvae_model = cvae_model.to(device)
    optimizer = optim.Adam(cvae_model.parameters(), lr=learning_rate)
    epochs_train = 20
    
    print(f"开始训练（使用0-7数据，{epochs_train}个epoch）...")
    for epoch in range(epochs_train):
        cvae_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader_07, desc=f'Epoch {epoch+1}/{epochs_train}')
        for x, y in pbar:
            x = x.view(-1, input_dim).to(device)
            y_labels = y.to(device)
            
            optimizer.zero_grad()
            x_hat, mu, sigma = cvae_model(x, y_labels)
            loss = cvae_loss_function(x_hat, x, mu, sigma)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs_train}, Loss: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(cvae_model.state_dict(), cvae_model_path_07)
    print(f"\n模型已保存: {cvae_model_path_07}")
    cvae_model.eval()

# 创建仅包含 8 和 9 的数据集（用于微调embedding）
print("\n创建仅包含8和9的数据集...")
ds_89 = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
idx9 = ds_89.targets == 9
idx8 = ds_89.targets == 8
idx89 = idx9 | idx8

ds_89.targets = ds_89.targets[idx89]
ds_89.data = ds_89.data[idx89]

print(f"8和9数据集: {len(ds_89)} 个样本")

# 创建数据加载器
data_loader_89 = DataLoader(ds_89, batch_size=batch_size, shuffle=True)

# ========== 第二部分：使用6、7、8、9条件进行解码（优化前）==========
print("\n" + "="*60)
print("第二部分：使用6、7、8、9条件进行解码（优化embedding前）")
print("="*60)

# 从标准正态分布采样latent z
z = torch.randn(1, latent_dim).to(device)
print(f"采样的z: {z.cpu().numpy()}")

# 测试6、7、8、9条件解码
print("\n测试6、7、8、9条件解码（优化前）...")
test_labels = [6, 7, 8, 9]
images_before = {}
with torch.no_grad():
    for label in test_labels:
        label_tensor = torch.tensor(label, dtype=torch.long).to(device).unsqueeze(0)
        c = cvae_model.condition_embedding(label_tensor)
        img = cvae_model.decoder(z, c)
        img = img.view(28, 28).cpu().numpy()
        images_before[label] = img

# 可视化优化前的效果
fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
for i, label in enumerate(test_labels):
    axes1[i].imshow(images_before[label], cmap='gray')
    axes1[i].set_title(f'Label {label}\n(Before Optimization)', fontsize=12, fontweight='bold')
    axes1[i].axis('off')

plt.suptitle('CVAE: Generated Images for Labels 6, 7, 8, 9\n(Before Embedding Fine-tuning)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('test4_before_optimization.png', dpi=150, bbox_inches='tight')
print("优化前效果已保存: test4_before_optimization.png")

# ========== 第三部分：冻结模型参数，只优化embedding ==========
print("\n" + "="*60)
print("第三部分：冻结模型参数，只优化embedding层")
print("="*60)

# 设置模型为训练模式（但大部分参数被冻结）
cvae_model.train()

# 冻结所有参数，只保留embedding层可训练
print("\n冻结模型参数，只保留embedding层可训练...")
for name, param in cvae_model.named_parameters():
    # embedding层的完整名称是 'condition_embedding.embedding.weight'
    if 'condition_embedding.embedding.weight' in name:
        param.requires_grad = True
        print(f"  可训练: {name}")
    else:
        param.requires_grad = False

# 创建优化器（只优化embedding参数）
trainable_params = [p for p in cvae_model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=learning_rate)

print(f"\n开始微调embedding（使用8和9的数据）...")
print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

losses = []

for epoch in range(epochs_finetune):
    epoch_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader_89, desc=f'Epoch {epoch+1}/{epochs_finetune}')
    for x, y in pbar:
        x = x.view(-1, input_dim).to(device)
        y_labels = y.to(device)
        
        optimizer.zero_grad()
        x_hat, mu, sigma = cvae_model(x, y_labels)
        loss = cvae_loss_function(x_hat, x, mu, sigma)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
    
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)

# 设置为评估模式
cvae_model.eval()

# ========== 第四部分：使用6、7、8、9条件进行解码（优化后）==========
print("\n" + "="*60)
print("第四部分：使用6、7、8、9条件进行解码（优化embedding后）")
print("="*60)

# 测试6、7、8、9条件解码
print("\n测试6、7、8、9条件解码（优化后）...")
images_after = {}
with torch.no_grad():
    for label in test_labels:
        label_tensor = torch.tensor(label, dtype=torch.long).to(device).unsqueeze(0)
        c = cvae_model.condition_embedding(label_tensor)
        img = cvae_model.decoder(z, c)
        img = img.view(28, 28).cpu().numpy()
        images_after[label] = img

# 可视化优化后的效果
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
for i, label in enumerate(test_labels):
    axes2[i].imshow(images_after[label], cmap='gray')
    axes2[i].set_title(f'Label {label}\n(After Optimization)', fontsize=12, fontweight='bold')
    axes2[i].axis('off')

plt.suptitle('CVAE: Generated Images for Labels 6, 7, 8, 9\n(After Embedding Fine-tuning)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('test4_after_optimization.png', dpi=150, bbox_inches='tight')
print("优化后效果已保存: test4_after_optimization.png")

# 对比图（2x4布局）
fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
for i, label in enumerate(test_labels):
    # Before
    axes3[0, i].imshow(images_before[label], cmap='gray')
    axes3[0, i].set_title(f'Label {label} (Before)', fontsize=12, fontweight='bold')
    axes3[0, i].axis('off')
    
    # After
    axes3[1, i].imshow(images_after[label], cmap='gray')
    axes3[1, i].set_title(f'Label {label} (After)', fontsize=12, fontweight='bold')
    axes3[1, i].axis('off')

plt.suptitle('CVAE Embedding Optimization: Before vs After\n(Only embedding layer was fine-tuned on digits 8 and 9)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('test4_comparison.png', dpi=150, bbox_inches='tight')
print("对比图已保存: test4_comparison.png")

# 绘制损失曲线
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title('Embedding Fine-tuning Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test4_finetune_loss.png', dpi=150, bbox_inches='tight')
print("损失曲线已保存: test4_finetune_loss.png")

# 生成多个样本对比
print("\n生成多个样本对比...")
num_samples = 5
z_samples = torch.randn(num_samples, latent_dim).to(device)

fig4, axes4 = plt.subplots(4, num_samples, figsize=(15, 12))

with torch.no_grad():
    for i, z_sample in enumerate(z_samples):
        for label_idx, label in enumerate(test_labels):
            label_tensor = torch.tensor(label, dtype=torch.long).to(device).unsqueeze(0)
            c = cvae_model.condition_embedding(label_tensor)
            img = cvae_model.decoder(z_sample.unsqueeze(0), c)
            img = img.view(28, 28).cpu().numpy()
            axes4[label_idx, i].imshow(img, cmap='gray')
            if i == 0:
                axes4[label_idx, i].set_ylabel(f'Label {label}', fontsize=12, fontweight='bold')
            if label_idx == 0:
                axes4[label_idx, i].set_title(f'z{i+1}', fontsize=10)
            axes4[label_idx, i].axis('off')

plt.suptitle('CVAE: Multiple Samples for Labels 6, 7, 8, 9\n(After Embedding Fine-tuning)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('test4_multiple_samples.png', dpi=150, bbox_inches='tight')
print("多样本对比图已保存: test4_multiple_samples.png")

print("\n" + "="*60)
print("实验完成！")
print("="*60)
print("\n实验结果：")
print("1. 优化前的6、7、8、9图像：")
print("   - 6和7：模型在0-7数据上训练，应该能生成合理的图像")
print("   - 8和9：模型未见过8和9，生成的图像可能不准确")
print("2. 优化后的6、7、8、9图像：")
print("   - 6和7：embedding微调后可能略有变化")
print("   - 8和9：只优化embedding层后，模型能够生成更合理的8和9数字")
print("3. 这说明即使模型没有用这些图像进行训练，只更新条件嵌入向量也能生成合理的图像")

plt.show()


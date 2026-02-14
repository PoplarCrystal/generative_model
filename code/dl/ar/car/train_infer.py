"""CAR-RNN 训练与推理脚本（条件自回归，RNN 实现）"""
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

from car_rnn import CAR_RNN, loss_function

# 加载配置
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 模型参数
input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
dropout = config['model'].get('dropout', 0.2)
num_labels = config['model']['num_labels']
label_embed_dim = config['model'].get('label_embed_dim', hidden_dim)
seq_len = config['model']['seq_len']

# 训练参数
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
train = config['flags']['train']
infer = config['flags']['infer']

model_path = os.path.join(current_dir, config['model_file']['model_name'])
data_dir = os.path.normpath(
    os.path.join(current_dir, config['data']['data_dir'])
)
sample_size = config['visualization']['sample_size']
nrow = config['visualization']['nrow']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root=data_dir, train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root=data_dir, train=False, download=False, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

# 模型
model = CAR_RNN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    num_labels=num_labels,
    label_embed_dim=label_embed_dim,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []


def preprocess_images(images):
    """将图像转为像素序列"""
    images = (images > 0.5).float()
    return images.reshape(images.size(0), -1, 1)  # (B, 784, 1)


def prepare_sequences(pixel_seq):
    """准备输入与目标序列"""
    inputs = pixel_seq[:, :-1, :]
    targets = pixel_seq[:, 1:, :]
    return inputs, targets


if train:
    print('\n开始训练 CAR-RNN 模型...')
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pixel_seq = preprocess_images(images)
            inputs, targets = prepare_sequences(pixel_seq)

            logits = model(inputs, labels)
            loss = loss_function(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
            pbar.set_postfix({'loss': f'{loss_sum/cnt:.4f}'})

        train_loss_avg = loss_sum / cnt
        train_losses.append(train_loss_avg)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}')

    torch.save(model.state_dict(), model_path)
    print(f'模型已保存: {model_path}')

    # 测试
    print('\n开始测试...')
    model.eval()
    test_loss_sum = 0.0
    test_cnt = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pixel_seq = preprocess_images(images)
            inputs, targets = prepare_sequences(pixel_seq)
            logits = model(inputs, labels)
            loss = loss_function(logits, targets)
            test_loss_sum += loss.item()
            test_cnt += 1
            pbar.set_postfix({'test_loss': f'{test_loss_sum/test_cnt:.4f}'})
    print(f'Test Loss: {test_loss_sum / test_cnt:.4f}\n')

    if train_losses:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.axhline(
            y=test_loss_sum / test_cnt,
            color='r',
            linestyle='--',
            label=f'Test Loss: {test_loss_sum/test_cnt:.4f}',
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('CAR-RNN Training Loss')
        plt.show()

if not train:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'模型已加载: {model_path}')
    except FileNotFoundError:
        print(f'警告: 未找到模型 {model_path}，使用随机初始化')
    except Exception as e:
        print(f'加载模型出错: {e}')

if infer:
    model.eval()
    print('\n开始条件生成图像...')
    print('注意: AR 自回归逐像素生成，速度较慢...')

    with torch.no_grad():
        # 可指定 labels 或 None（随机 0-9）
        labels = torch.tensor(
            [i % 10 for i in range(sample_size)], device=device
        )  # 按 0,1,...,9 循环便于展示
        generated_sequences, labels = model.generate(
            batch_size=sample_size,
            seq_len=seq_len,
            labels=labels,
            device=device,
        )
        generated_images = (
            generated_sequences.squeeze(-1).view(sample_size, 1, 28, 28).cpu()
        )

    # 与 CVAE/CGAN/CDDPM 一致：每张图上方显示 Label，做到 label 与图片一一对应
    fig, axes = plt.subplots(nrow, nrow, figsize=(16, 16))
    fig.suptitle('Generated Images (CAR-RNN, by label)', fontsize=16, y=0.995)
    labels_cpu = labels.cpu()
    for idx, ax in enumerate(axes.flat):
        if idx < sample_size:
            ax.imshow(generated_images[idx].squeeze().numpy(), cmap='gray')
            ax.set_title(f'Label: {labels_cpu[idx].item()}', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.2, wspace=0.1)
    plt.show()

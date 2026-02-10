import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torchvision import datasets, transforms
from tqdm import tqdm
from cvae import CVAE, ConditionEncoder

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def estimate_log_probability(model, x, c_label, n_samples=100):
    """
    使用重要性采样 (Importance Sampling) 估计 log p(x|c)
    log p(x|c) ≈ log(1/K * Σ [p(x|z,c)p(z)/q(z|x,c)])
    """
    model.eval()
    with torch.no_grad():
        # 1. 准备数据
        # x: (1, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 准备条件 c
        label_tensor = torch.tensor([c_label], device=x.device)
        if model.condition_encoder is not None:
            c = model.condition_encoder(label_tensor)
        else:
            c = label_tensor
            
        # 2. 编码 q(z|x,c)
        x_c = torch.cat((x, c), dim=1)
        mu_q, sigma_q = model.encoder(x_c)
        
        # 3. 准备多次采样
        # 扩展维度: (n_samples, 1, dim)
        mu_q = mu_q.expand(n_samples, -1, -1)
        sigma_q = sigma_q.expand(n_samples, -1, -1)
        
        c_expanded = c.expand(n_samples, -1, -1)
        x_expanded = x.expand(n_samples, -1, -1)
        
        # 采样 z ~ q(z|x,c)
        epsilon = torch.randn_like(mu_q)
        z = mu_q + epsilon * sigma_q
        
        # 4. 计算 log 概率
        
        # log q(z|x,c)
        log_q_z = -0.5 * torch.sum(
            np.log(2 * np.pi) + 2 * torch.log(sigma_q + 1e-8) + epsilon ** 2, 
            dim=-1
        )
        
        # log p(z) - 标准正态先验 N(0, I)
        log_p_z = -0.5 * torch.sum(
            np.log(2 * np.pi) + z ** 2, 
            dim=-1
        )
        
        # log p(x|z,c) - 解码器似然
        z_c = torch.cat((z, c_expanded), dim=-1)
        x_hat = model.decoder(z_c)
        
        # 假设输出分布为高斯 (MSE Loss)
        # 注意：这里直接使用 MSE 作为 log p(x|z,c) 的代理
        # 对于二值化 MNIST，如果输出层是 Sigmoid，可以用 BCE
        # 这里沿用 CVAE 的 MSE 设定
        recon_error = torch.sum((x_expanded - x_hat) ** 2, dim=-1)
        log_p_x_z = -recon_error
        
        # 5. 计算 Importance Weights
        log_w = log_p_x_z + log_p_z - log_q_z
        
        # 6. Log-Sum-Exp
        log_prob = torch.logsumexp(log_w, dim=0) - np.log(n_samples)
        
        return log_prob.item()

def classify_sample(model, x, num_classes, verbose=False):
    """
    使用 CVAE 进行分类：ArgMax p(x|c)
    """
    log_probs = []
    
    for c in range(num_classes):
        log_prob = estimate_log_probability(model, x, c)
        log_probs.append(log_prob)
        
    predicted_label = np.argmax(log_probs)
    return predicted_label, log_probs

def main():
    # 1. 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')
    
    print(f"读取配置: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 从配置文件读取参数
    input_dim = config['model']['input_dim']
    output_dim = config['model']['output_dim']
    n_classes = config['model']['n_classes']
    condition_encoding_type = config['model'].get('condition_encoding_type', 'onehot')
    embed_dim = config['model'].get('embed_dim', None)
    hidden_dim = config['model']['hidden_dim']
    latent_dim = config['model']['latent_dim']
    data_dir_rel = config['data']['data_dir']
    
    # 确定模型路径
    model_name = config['model_file']['model_name']
    if not model_name:
        model_name = 'cvae_model.pth'
    model_path = os.path.join(current_dir, model_name)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    
    # 2. 准备数据 (MNIST Test)
    data_dir = os.path.join(current_dir, data_dir_rel)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    
    # 3. 初始化并加载模型
    print(f"加载模型: {model_path}")
    
    condition_encoder = ConditionEncoder(
        n_classes=n_classes,
        condition_encoding_type=condition_encoding_type,
        embed_dim=embed_dim
    ).to(device)
    
    condition_dim = condition_encoder.condition_dim
    
    model = CVAE(
        input_dim, output_dim, condition_dim, hidden_dim, latent_dim,
        condition_encoder=condition_encoder
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}。请先运行 train_infer.py 进行训练。")
        return

    # 4. 进行判别测试
    # 获取一个 batch (16张图)
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    images = images.view(-1, input_dim).to(device)
    labels = labels.to(device)
    
    predictions = []
    print(f"正在对 {len(images)} 张 MNIST 测试图片进行分类...")
    
    for i in tqdm(range(len(images))):
        x = images[i]
        pred_label, _ = classify_sample(model, x, n_classes)
        predictions.append(pred_label)
        
    # 5. 可视化
    nrow = 4
    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    fig.suptitle('MNIST Classification using CVAE Density Estimation\n(Pred vs True)', fontsize=16)
    
    images_cpu = images.cpu().view(-1, 28, 28)
    labels_cpu = labels.cpu().numpy()
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images_cpu[idx].numpy()
            true_lbl = labels_cpu[idx]
            pred_lbl = predictions[idx]
            
            ax.imshow(img, cmap='gray')
            
            color = 'green' if true_lbl == pred_lbl else 'red'
            
            # 使用 title 替代 text，避免遮挡
            ax.set_title(f'True: {true_lbl} | Pred: {pred_lbl}', 
                        fontsize=10, color=color, fontweight='bold', pad=5)
            
        ax.axis('off')
    
    plt.tight_layout()
    # 调整顶部边距以容纳总标题
    plt.subplots_adjust(top=0.90, hspace=0.4)
    
    save_path = 'mnist_classification_results.png'
    plt.savefig(save_path)
    print(f"\n结果已保存至: {save_path}")

if __name__ == "__main__":
    main()

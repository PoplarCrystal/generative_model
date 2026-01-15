# AE、VAE、CVAE 潜在空间可视化对比

本项目实现了**AE（自编码器）**、**VAE（变分自编码器）**、**CVAE（条件变分自编码器）**和**CVAE_prior**四个模型，使用`latent_dim=2`进行2D潜在空间可视化，用于对比不同模型如何将MNIST数字映射到潜在空间。

## 📋 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [文件说明](#文件说明)
- [快速开始](#快速开始)
- [实验脚本](#实验脚本)
- [输出文件](#输出文件)
- [模型架构](#模型架构)

## 📖 项目概述

本项目旨在通过2D可视化直观理解不同自编码器模型的潜在空间特性：

1. **潜在空间的分布**：对比AE和VAE的潜在空间分布差异
2. **正则化的意义**：理解KL散度正则化如何影响潜在空间的生成能力
3. **条件的影响**：探索条件信息如何改变潜在空间的分布
4. **编码的影响**：研究条件编码方式（embedding）如何影响模型的扩展能力

## 🔧 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- tqdm

## 📁 文件说明

### 核心文件

- **`models.py`**: 包含所有模型的定义
  - `AE`: 自编码器
  - `VAE`: 变分自编码器
  - `CVAE`: 条件变分自编码器（c与z相互独立）
  - `CVAE_prior`: 条件变分自编码器（c与z非独立）
  - 对应的损失函数

- **`train_and_visualize.py`**: 主训练脚本
  - 训练所有模型（如果模型文件不存在）
  - 生成2D潜在空间可视化
  - 生成对比图

### 实验脚本

- **`test1_ae_vae.py`**: 对比AE和VAE的解码器生成能力
  - 从标准正态分布采样latent z
  - 分别使用AE和VAE解码器生成图像
  - 对比生成结果

- **`test2_vae.py`**: VAE潜在空间插值动画
  - 获取数字1和数字8的图片
  - 通过VAE编码器得到z1和z8
  - 生成插值动画：z = α·z1 + (1-α)·z8，α从1到0

- **`test3_cvae.py`**: CVAE条件渐变实验
  - 固定潜在向量z
  - 条件c从0到9渐变（使用embedding编码）
  - 观察生成的图像变化动画

- **`test4_cvae_embed.py`**: CVAE Embedding优化实验
  - 使用0-7的数据训练CVAE模型
  - 冻结模型参数，只优化embedding层
  - 使用8和9的数据微调embedding
  - 验证能否生成合理的8和9数字

## 🚀 快速开始

### 1. 训练模型并生成可视化

```bash
cd generative_model/code/dl/ae/ae_all
python train_and_visualize.py
```

**注意**：如果模型文件（`.pth`）已存在，脚本会自动加载，不会重新训练。

### 2. 运行实验脚本

```bash
# 测试1: AE vs VAE 解码器对比
python test1_ae_vae.py

# 测试2: VAE潜在空间插值动画
python test2_vae.py

# 测试3: CVAE条件渐变
python test3_cvae.py

# 测试4: CVAE Embedding优化
python test4_cvae_embed.py
```

## 📊 输出文件

### `train_and_visualize.py` 生成的文件

#### 模型文件
- `ae_model.pth` - 训练好的AE模型
- `vae_model.pth` - 训练好的VAE模型
- `cvae_model.pth` - 训练好的CVAE模型
- `cvae_prior_model.pth` - 训练好的CVAE_prior模型

#### 可视化文件
- `ae_latent_space.png` - AE的2D潜在空间散点图
- `vae_latent_space.png` - VAE的2D潜在空间散点图（使用μ）
- `cvae_latent_space.png` - CVAE的2D潜在空间散点图（使用μ）
- `cvae_prior_latent_space.png` - CVAE_prior的2D潜在空间散点图（使用μ）

#### 对比图
- `comparison_ae_vae.png` - AE vs VAE对比图
- `comparison_vae_cvae.png` - VAE vs CVAE对比图
- `comparison_cvae_cvae_prior.png` - CVAE vs CVAE_prior对比图
- `comparison_all_four.png` - 四个模型的2x2对比图

#### 训练曲线
- `training_losses.png` - 训练损失曲线对比

### 实验脚本生成的文件

#### test1_ae_vae.py
- `test1_ae_vae_comparison.png` - AE和VAE解码器生成图像对比

#### test2_vae.py
- `test2_vae_interpolation.gif` - VAE潜在空间插值动画
- `test2_vae_keyframes.png` - 插值关键帧

#### test3_cvae.py
- `test3_cvae_condition_interpolation.gif` - CVAE条件渐变动画
- `test3_cvae_keyframes.png` - 条件渐变关键帧
- `test3_cvae_gradient_bar.png` - 渐变条可视化
- `test3_cvae_different_z.png` - 不同z值的对比图

#### test4_cvae_embed.py
- `cvae_model_0_7.pth` - 只使用0-7训练得到的CVAE模型
- `test4_before_optimization.png` - 优化embedding前的生成效果
- `test4_after_optimization.png` - 优化embedding后的生成效果
- `test4_comparison.png` - 优化前后对比图
- `test4_finetune_loss.png` - Embedding微调损失曲线
- `test4_multiple_samples.png` - 多个样本的生成结果

## 🏗️ 模型架构

### AE (Autoencoder)
- **编码器**：直接输出潜在向量$z$
- **解码器**：从$z$重构输入
- **损失函数**：重构损失 $\mathcal{L} = \|\hat{x} - x\|^2$
- **特点**：无正则化，潜在空间分布任意

### VAE (Variational Autoencoder)
- **编码器**：输出分布参数$(\mu, \sigma)$
- **重参数化**：$z = \mu + \epsilon \cdot \sigma$，其中$\epsilon \sim \mathcal{N}(0, I)$
- **解码器**：从$z$生成输出
- **损失函数**：重构损失 + KL散度正则化
  $$\mathcal{L} = \|\hat{x} - x\|^2 + \beta \cdot \text{KL}(q(z|x) \| \mathcal{N}(0, I))$$
- **特点**：通过KL散度将潜在空间约束到标准正态分布

### CVAE (Conditional VAE)
- **编码器**：输入$[x, c]$，输出$(\mu, \sigma)$
- **解码器**：输入$[z, c]$，生成输出
- **先验分布**：$p(z|c) = p(z) = \mathcal{N}(0, I)$（条件与z相互独立）
- **特点**：条件信息控制生成过程，但先验分布不受条件影响

### CVAE_prior
- **编码器**：输入$[x, c]$，输出$(\mu, \sigma)$
- **先验网络**：输入$c$，输出先验分布参数$(\mu_p, \sigma_p)$
- **解码器**：输入$[z, c]$，生成输出
- **先验分布**：$p(z|c) = \mathcal{N}(\mu_p(c), \sigma_p(c))$（条件依赖）
- **特点**：条件不仅影响编码器和解码器，还影响先验分布

## 📈 可视化说明

### 散点图颜色编码
- 不同颜色代表不同的数字标签（0-9）
- 颜色从深到浅表示0到9（使用viridis色彩映射）

### 主要观察点
- **AE**：潜在空间可能无序，存在空洞，无法从标准正态分布采样生成合理图像
- **VAE**：潜在空间被正则化，接近标准正态分布，具有强大的生成能力
- **CVAE**：条件信息可以控制生成过程，相同z在不同条件下生成不同类别的图像
- **CVAE_prior**：先验分布也依赖于条件，可能提供更好的条件控制


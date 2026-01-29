# Generative Models

生成模型相关代码和文档集合。本项目涵盖了从基础的机器学习方法到深度学习的各种生成模型，包括自编码器系列、扩散模型系列、生成对抗网络等。

## 目录结构

```
generative_model/
├── code/                    # 代码实现
│   ├── dl/                  # 深度学习相关
│   │   ├── ae/             # 自编码器系列
│   │   │   ├── ae/        # 基础自编码器
│   │   │   ├── ae_all/    # 自编码器集合（AE, VAE, CVAE）
│   │   │   ├── vae/       # 变分自编码器
│   │   │   ├── cvae/      # 条件变分自编码器
│   │   │   ├── vae_cg/    # VAE with Classifier Guidance
│   │   │   └── vq_vae/    # 向量量化变分自编码器
│   │   ├── dm/            # 扩散模型系列
│   │   │   ├── ddpm/      # Denoising Diffusion Probabilistic Models
│   │   │   ├── ddim/      # Denoising Diffusion Implicit Models
│   │   │   ├── cddpm/     # Conditional DDPM
│   │   │   ├── ddpm_cg/   # DDPM with Classifier Guidance
│   │   │   ├── ddpm_cfg/  # DDPM with Classifier-Free Guidance
│   │   │   ├── ldm/       # Latent Diffusion Model
│   │   │   └── ...        # 其他变体
│   │   └── gan/           # 生成对抗网络
│   │       ├── gan/       # 基础GAN
│   │       ├── cgan/      # Conditional GAN
│   │       └── mode.py    # Mode Covering vs Mode Seeking 可视化
│   ├── ml/                # 机器学习相关
│   │   ├── 02/           # 最大似然估计（MLE）
│   │   ├── 03/           # 高斯混合模型（GMM）
│   │   └── 04/           # 期望最大化算法（EM）
│   └── data/              # 数据目录
│       └── MNIST/        # MNIST数据集
├── pic/                   # 相关图片和可视化结果
│   ├── dl/               # 深度学习相关图片
│   └── ml/               # 机器学习相关图片
└── *.md                   # 文档和笔记
```

## 代码说明

### 深度学习模型（code/dl/）

#### 自编码器系列（ae/）

- **ae/** - 基础自编码器（Autoencoder）
  - `ae.py` - 自编码器模型
  - `train_infer.py` - 训练和推理脚本
  - `visualize_compression.py` - 压缩可视化
  - `visualize_svd.py` - SVD可视化

- **vae/** - 变分自编码器（Variational Autoencoder）
  - `vae.py` - VAE模型实现
  - `train_infer.py` - 训练和推理

- **cvae/** - 条件变分自编码器（Conditional VAE）
  - `cvae.py` - CVAE模型实现
  - `train_infer.py` - 训练和推理

- **vae_cg/** - VAE with Classifier Guidance
  - `vae_cg.py` - VAE-CG模型实现

- **vq_vae/** - 向量量化变分自编码器（VQ-VAE）
  - `vq_vae.py` - VQ-VAE模型实现
  - `pixelcnn_model.py` - PixelCNN先验模型
  - `train_infer.py` - 训练和推理

#### 扩散模型系列（dm/）

- **ddpm/** - Denoising Diffusion Probabilistic Models
  - `ddpm.py` - DDPM模型实现
  - `unet.py` - U-Net架构
  - `train_infer.py` - 训练和推理

- **ddim/** - Denoising Diffusion Implicit Models
  - `ddim.py` - DDIM采样器实现
  - `train_infer.py` - 训练和推理

- **cddpm/** - Conditional DDPM
  - `cddpm.py` - 条件扩散模型实现
  - `unet.py` - 条件U-Net架构

- **ddpm_cg/** - DDPM with Classifier Guidance
  - 带分类器引导的DDPM实现

- **ddpm_cfg/** - DDPM with Classifier-Free Guidance
  - 无分类器引导的DDPM实现

- **ldm/** - Latent Diffusion Model
  - `ldm.py` - 潜在扩散模型实现
  - `vae.py` - VAE编码器/解码器
  - `unet.py` - 潜在空间U-Net
  - `train_infer.py` - 训练和推理

#### 生成对抗网络（gan/）

- **gan/** - 基础生成对抗网络
  - `gan.py` - GAN模型实现
  - `train_infer.py` - 训练和推理

- **cgan/** - Conditional GAN
  - `cgan.py` - 条件GAN实现

- **mode.py** - Mode Covering vs Mode Seeking 可视化实验

### 机器学习模型（code/ml/）

- **02/** - 最大似然估计（MLE）
  - `fit.py` - 参数拟合
  - `generate.py` - 数据生成
  - `hist.py` - 直方图可视化

- **03/** - 高斯混合模型（GMM）
  - `gmm.py` - GMM实现
  - `gmm_sampling.py` - GMM采样
  - `old_faithful.py` - 老忠实间歇泉数据示例

- **04/** - 期望最大化算法（EM）
  - `em.py` - EM算法实现
  - `generate.py` - 数据生成

## 文档

### 机器学习基础篇

- `机器学习篇-01生成模型基础.md` - 生成模型基础理论
- `机器学习篇-02最大似然估计.md` - 最大似然估计方法
- `机器学习篇-03潜变量估计.md` - 潜变量估计方法
- `机器学习篇-04变分推断.md` - 变分推断理论


### AE系列篇

- `AE系列篇-01AE.md` - 自编码器基础
- `AE系列篇-02VAE理论篇.md` - 变分自编码器理论
- `AE系列篇-02VAE代码篇.md` - 变分自编码器代码实现
- `AE系列篇-03CVAE理论篇.md` - 条件变分自编码器理论
- `AE系列篇-03CVAE代码篇.md` - 条件变分自编码器代码实现
- `AE系列篇-04潜变量的可解释性.md` - 潜变量可解释性分析
- `AE系列篇-05VAE-CG理论篇.md` - VAE with Classifier Guidance理论

### DM系列篇（扩散模型）

- `DM系列篇-01DPM.md` - 扩散概率模型基础
- `DM系列篇-02不同形式的DPM.md` - DPM的不同形式
- `DM系列篇-03.DDPM实现.md` - DDPM实现详解
- `DM系列篇-04.DPM三种实现对比.md` - DPM三种实现方式对比
- `DM系列篇-05.CDDPM理论篇1.md` - 条件DDPM理论（上）
- `DM系列篇-06.CDDPM理论篇2.md` - 条件DDPM理论（下）
- `DM系列篇-07. CDDPM实现篇1.md` - 条件DDPM实现（上）
- `DM系列篇-08. CDDPM实现篇2.md` - 条件DDPM实现（下）
- `DM系列篇-09. DDPM的改进1：DDIM.md` - DDIM加速采样
- `DM系列篇-10. DDPM的改进2：LDM.md` - 潜在扩散模型

### 补充系列篇

- `补充系列篇-01Mode.md` - Mode Covering vs Mode Seeking 详解

### 其他文档

- `VQ-VAE_算法推导.md` - VQ-VAE算法详细推导

## 环境安装

### 快速安装（推荐）

**Linux/macOS:**
```bash
bash setup_env.sh
```

### 手动安装

详细安装步骤请参考 [INSTALL.md](INSTALL.md)（如果存在）

### 环境要求

- Python 3.9+
- PyTorch 1.12+
- NumPy
- Matplotlib
- torchvision
- tqdm
- scipy
- yaml

## 使用方法

### 激活环境

```bash
conda activate gen_model
# 或
source activate gen_model
```

### 自编码器系列

```bash
# 基础自编码器
cd code/dl/ae/ae
python train_infer.py

# 变分自编码器
cd code/dl/vae
python train_infer.py

# 条件变分自编码器
cd code/dl/cvae
python train_infer.py

# VQ-VAE
cd code/dl/vq_vae
python train_infer.py
```

### 扩散模型系列

```bash
# DDPM
cd code/dl/dm/ddpm
python train_infer.py

# DDIM
cd code/dl/dm/ddim
python train_infer.py

# Conditional DDPM
cd code/dl/dm/cddpm
python train_infer.py

# Latent Diffusion Model
cd code/dl/dm/ldm
python train_infer.py
```

### 生成对抗网络

```bash
# 基础GAN
cd code/dl/gan/gan
python train_infer.py

# Conditional GAN
cd code/dl/gan/cgan
python train_infer.py

# Mode Covering vs Mode Seeking 可视化
cd code/dl/gan
python mode.py
```

### 机器学习模型

```bash
# 最大似然估计
cd code/ml/02
python fit.py

# 高斯混合模型
cd code/ml/03
python gmm.py

# EM算法
cd code/ml/04
python em.py
```

## 项目特点

- **完整的理论文档**：从基础理论到代码实现，详细讲解各种生成模型
- **丰富的代码实现**：涵盖AE、VAE、DDPM、DDIM、LDM、GAN等多种模型
- **可视化实验**：包含多种可视化工具和实验代码
- **清晰的代码结构**：每个模型都有独立的目录和配置文件

## License

MIT License

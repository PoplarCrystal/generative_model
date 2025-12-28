# Generative Models

生成模型相关代码和文档集合。

## 目录结构

```
generative_model/
├── code/           # 代码实现
│   ├── dl/        # 深度学习相关（VAE, CVAE, AE等）
│   └── ml/        # 机器学习相关（MLE, GMM, EM等）
├── pic/           # 相关图片
└── *.md           # 文档和笔记
```

## 代码说明

### 深度学习模型（code/dl/）

- **ae.py** - 自编码器（Autoencoder）
- **vae.py** - 变分自编码器（Variational Autoencoder）
- **vae_cnn.py** - 卷积变分自编码器
- **cvae.py** - 条件变分自编码器（Conditional VAE）
- **cvae1.py** - 条件变分自编码器变体
- **vq_vae.py** - 向量量化变分自编码器（VQ-VAE）

### 机器学习模型（code/ml/）

- **02/** - 最大似然估计（MLE）
- **03/** - 高斯混合模型（GMM）
- **04/** - 期望最大化算法（EM）

## 文档

- `机器学习篇-01生成模型基础.md` - 生成模型基础
- `机器学习篇-02最大似然估计.md` - 最大似然估计
- `机器学习篇-03潜变量估计.md` - 潜变量估计
- `机器学习篇-04变分推断.md` - 变分推断

## 环境安装

### 快速安装（推荐）

**Linux/macOS:**
```bash
bash setup_env.sh
```

**Windows:**
```cmd
setup_env.bat
```

### 手动安装

详细安装步骤请参考 [INSTALL.md](INSTALL.md)

### 环境要求

- Python 3.9+
- PyTorch 1.12+
- NumPy
- Matplotlib
- torchvision
- tqdm

## 使用方法

```bash
# 激活环境
conda activate gen_model

# 训练模型
python code/dl/vae.py
python code/dl/vq_vae.py

# 测试模型
python code/dl/ae.py
```

## License

MIT License



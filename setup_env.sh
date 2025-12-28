#!/bin/bash

# 生成模型项目 Conda 环境安装脚本
# 使用方法: bash setup_env.sh

ENV_NAME="gen_model"
PYTHON_VERSION="3.9"

echo "=========================================="
echo "创建 Conda 环境: $ENV_NAME"
echo "=========================================="

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "警告: 环境 $ENV_NAME 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "使用现有环境"
        conda activate $ENV_NAME
        pip install -r requirements.txt
        echo "环境设置完成！"
        exit 0
    fi
fi

# 创建新的 conda 环境
echo "创建 Python $PYTHON_VERSION 环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
echo "激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 检查 CUDA 是否可用（用于决定安装哪个版本的 PyTorch）
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU，将安装支持 CUDA 的 PyTorch"
    CUDA_AVAILABLE=true
else
    echo "未检测到 NVIDIA GPU，将安装 CPU 版本的 PyTorch"
    CUDA_AVAILABLE=false
fi

# 安装 PyTorch（根据 CUDA 版本选择）
echo "=========================================="
echo "安装 PyTorch..."
echo "=========================================="

if [ "$CUDA_AVAILABLE" = true ]; then
    # 检测 CUDA 版本
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "检测到 CUDA 版本: $CUDA_VERSION"
    
    # 根据 CUDA 版本安装 PyTorch（默认使用 11.8，兼容性最好）
    echo "安装 PyTorch (CUDA 11.8+)..."
    echo "注意: 如果您的 CUDA 版本较低，可能需要手动安装对应版本"
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "安装 PyTorch (CPU 版本)..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# 安装其他依赖
echo "=========================================="
echo "安装其他依赖包..."
echo "=========================================="

conda install matplotlib numpy -y
pip install tqdm

# 如果存在 requirements.txt，也安装其中的包
if [ -f "requirements.txt" ]; then
    echo "从 requirements.txt 安装额外依赖..."
    pip install -r requirements.txt
fi

echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo ""
echo "激活环境: conda activate $ENV_NAME"
echo "验证安装: python -c 'import torch; print(torch.__version__)'"
echo ""


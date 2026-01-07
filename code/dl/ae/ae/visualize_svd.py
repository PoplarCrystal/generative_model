"""
可视化脚本：展示10×10矩阵A的SVD分解
生成对比图：A矩阵 | SVD分解 | 前5个奇异值
"""
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# 定义10×10矩阵A
A = np.array([
    [1.2, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
    [0.8, 1.5, 1.1, 0.7, 0.5, 0.3, 0.15, 0.08, 0.03, 0.01],
    [0.6, 1.1, 1.8, 1.2, 0.8, 0.5, 0.25, 0.12, 0.05, 0.02],
    [0.4, 0.7, 1.2, 1.6, 1.0, 0.6, 0.3, 0.15, 0.06, 0.02],
    [0.3, 0.5, 0.8, 1.0, 1.4, 0.8, 0.4, 0.2, 0.08, 0.03],
    [0.2, 0.3, 0.5, 0.6, 0.8, 1.2, 0.6, 0.3, 0.12, 0.04],
    [0.1, 0.15, 0.25, 0.3, 0.4, 0.6, 1.0, 0.5, 0.2, 0.08],
    [0.05, 0.08, 0.12, 0.15, 0.2, 0.3, 0.5, 0.8, 0.3, 0.12],
    [0.02, 0.03, 0.05, 0.06, 0.08, 0.12, 0.2, 0.3, 0.6, 0.25],
    [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.08, 0.12, 0.25, 0.5]
])

# 进行SVD分解
U, s, Vt = np.linalg.svd(A, full_matrices=True)
# s是奇异值向量，需要转换为对角矩阵
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, s)
V = Vt.T

# 获取前5个奇异值
top5_singular_values = s[:5]

# 创建可视化 - 三个子图，增大尺寸并减小间距
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# 左：A矩阵（显示数字）
axes[0].axis('off')
axes[0].set_title('Matrix A (10×10)', fontsize=16, fontweight='bold', pad=10)
# 创建文本表格显示矩阵
A_text = ''
for i in range(10):
    row = ' '.join([f'{A[i, j]:5.2f}' for j in range(10)])
    A_text += row + '\n'
axes[0].text(0.5, 0.5, A_text, transform=axes[0].transAxes,
            fontsize=9, family='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# 中：SVD分解（显示公式和所有奇异值）
axes[1].axis('off')
axes[1].set_title('SVD Decomposition: A = UΣV^T', fontsize=16, fontweight='bold', pad=10)

# 显示SVD分解的关键信息和所有奇异值
svd_text = 'A = U × Σ × V^T\n\n'
svd_text += 'All 10 Singular Values:\n'
for i, sigma in enumerate(s, 1):
    svd_text += f'  σ{i} = {sigma:.3f}\n'

axes[1].text(0.5, 0.5, svd_text, transform=axes[1].transAxes,
            fontsize=13, family='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# 右：前5个奇异值（显示数字）
axes[2].axis('off')
axes[2].set_title('Top 5 Singular Values', fontsize=16, fontweight='bold', pad=10)
# 将前5个奇异值格式化显示
sigma_text = ''
for i, sigma in enumerate(top5_singular_values, 1):
    sigma_text += f'σ{i} = {sigma:.3f}\n'
axes[2].text(0.5, 0.5, sigma_text, transform=axes[2].transAxes,
            fontsize=18, family='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout(pad=1.0)  # 减小整体边距
plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.95, bottom=0.05)  # 减小间距，增大内容区域

# 保存图片
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'svd_decomposition_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Visualization saved: {output_path}')

plt.show()


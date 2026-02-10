# VQ-VAE 算法推导与公式

## 1. 背景与动机

### 1.1 VAE 的局限性

标准 VAE 使用**连续潜在变量** $z$，其分布为：

$$q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \Sigma_\phi(x))$$

**问题**：
- 连续潜在空间可能无法很好地表示离散的、多模态的数据结构
- 对于符号数据（如语言、类别）缺乏自然的表示

### 1.2 离散潜在变量的优势

**为什么需要离散潜在变量？**

1. **建模多峰分布**：离散表示可以更好地捕获数据的多个模式
2. **符号表示**：适合语言、语音、规划等符号系统
3. **信息瓶颈**：离散表示提供有界的信息容量，有助于学习有意义的表示

### 1.3 VQ-VAE 的核心思想

VQ-VAE（Vector Quantized Variational Autoencoder）将连续潜在变量**量化**为离散的 codebook 向量，从而结合了 VAE 的表达能力和离散表示的优点。

---

## 2. VQ-VAE 架构

### 2.1 整体结构

```
x → [编码器] → z_e (连续) → [向量量化] → z_q (离散) → [解码器] → x̂
```

**组件说明**：
- **编码器** $E_\phi$：$x \to z_e$，将输入映射到连续潜在空间
- **Codebook** $\mathcal{E} = \{e_1, e_2, \ldots, e_K\}$：包含 $K$ 个可学习的嵌入向量
- **向量量化** $Q$：$z_e \to z_q$，将连续向量映射到最近的 codebook 向量
- **解码器** $D_\theta$：$z_q \to x̂$，从量化向量重构输入

### 2.2 数学表示

**编码过程**：
$$z_e = E_\phi(x) \in \mathbb{R}^d$$

**量化过程**：
$$z_q = Q(z_e) = e_k, \quad \text{其中 } k = \arg\min_j \|z_e - e_j\|^2$$

**解码过程**：
$$x̂ = D_\theta(z_q)$$

---

## 3. 向量量化（Vector Quantization）

### 3.1 最近邻查找

给定编码器输出 $z_e$ 和 codebook $\mathcal{E} = \{e_1, e_2, \ldots, e_K\}$，量化操作选择最近的 codebook 向量：

$$k = \arg\min_{j \in \{1, \ldots, K\}} \|z_e - e_j\|^2$$

$$z_q = e_k$$

### 3.2 距离计算

使用欧氏距离的平方：

$$\|z_e - e_j\|^2 = (z_e - e_j)^T(z_e - e_j)$$

展开后：

$$\|z_e - e_j\|^2 = z_e^T z_e - 2 z_e^T e_j + e_j^T e_j$$

在实现中，可以预先计算 $e_j^T e_j$ 以提高效率。

---

## 4. 损失函数推导

### 4.1 目标函数

VQ-VAE 的目标是最小化重构误差，同时学习有意义的 codebook 表示。

**总损失函数**：

$$\mathcal{L}_{VQ-VAE} = \mathcal{L}_{recon} + \mathcal{L}_{VQ}$$

### 4.2 重构损失（Reconstruction Loss）

重构损失衡量解码器重构输入的能力：

$$\mathcal{L}_{recon} = \|x - D_\theta(z_q)\|^2$$

对于图像数据，通常使用：

$$\mathcal{L}_{recon} = \|x - x̂\|^2 = \sum_{i=1}^{N} (x_i - x̂_i)^2$$

其中 $N$ 是像素数量。

### 4.3 VQ 损失（Vector Quantization Loss）

VQ 损失包含两个部分：

#### 4.3.1 Codebook 损失（Codebook Loss）

将 codebook 向量 $e_k$ 拉近编码器输出 $z_e$：

$$\mathcal{L}_{codebook} = \|z_e - \text{sg}(z_q)\|^2$$

其中 $\text{sg}(\cdot)$ 表示停止梯度（stop gradient），即在前向传播中使用 $z_q$，但在反向传播中不计算其梯度。

**梯度更新**：
$$\frac{\partial \mathcal{L}_{codebook}}{\partial e_k} = 2(z_e - e_k)$$

这会将 codebook 向量 $e_k$ 向编码器输出 $z_e$ 移动。

#### 4.3.2 Commitment 损失（Commitment Loss）

将编码器输出 $z_e$ 拉近量化向量 $z_q$，防止编码器输出在 codebook 向量之间波动：

$$\mathcal{L}_{commitment} = \beta \|\text{sg}(z_q) - z_e\|^2$$

其中 $\beta$ 是 commitment 权重（通常取 0.25）。

**梯度更新**：
$$\frac{\partial \mathcal{L}_{commitment}}{\partial z_e} = 2\beta(z_e - z_q)$$

这会将编码器输出 $z_e$ 向量化向量 $z_q$ 移动。

#### 4.3.3 总 VQ 损失

$$\mathcal{L}_{VQ} = \mathcal{L}_{codebook} + \mathcal{L}_{commitment}$$

$$= \|z_e - \text{sg}(z_q)\|^2 + \beta \|\text{sg}(z_q) - z_e\|^2$$

注意：$\|z_e - \text{sg}(z_q)\|^2 = \|\text{sg}(z_q) - z_e\|^2$，所以：

$$\mathcal{L}_{VQ} = (1 + \beta) \|z_e - z_q\|^2$$

但在实际实现中，两个项使用不同的停止梯度位置，以控制不同组件的梯度流。

---

## 5. 直通估计器（Straight-Through Estimator）

### 5.1 问题

量化操作 $Q(z_e) = e_k$ 是不可微的，因为 $\arg\min$ 操作是离散的。这导致无法通过量化层反向传播梯度。

### 5.2 解决方案：直通技巧

**直通估计器**允许梯度通过不可微的量化操作：

**前向传播**：
$$z_q = Q(z_e) = e_k \quad \text{(硬量化)}$$

**反向传播**：
$$\frac{\partial \mathcal{L}}{\partial z_e} = \frac{\partial \mathcal{L}}{\partial z_q} \quad \text{(直通梯度)}$$

**实现方式**：
$$z_q^{ST} = z_q + (z_e - \text{sg}(z_e))$$

或者等价地：
$$z_q^{ST} = z_e + (z_q - \text{sg}(z_e))$$

这样在前向传播中使用 $z_q$，但在反向传播中梯度直接流向 $z_e$。

### 5.3 数学证明

考虑损失函数对 $z_e$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial z_e} = \frac{\partial \mathcal{L}}{\partial z_q^{ST}} \cdot \frac{\partial z_q^{ST}}{\partial z_e}$$

由于 $z_q^{ST} = z_q + (z_e - \text{sg}(z_e))$，且 $\text{sg}(z_e)$ 的梯度为 0：

$$\frac{\partial z_q^{ST}}{\partial z_e} = \frac{\partial}{\partial z_e}[z_q + z_e - \text{sg}(z_e)] = I$$

因此：

$$\frac{\partial \mathcal{L}}{\partial z_e} = \frac{\partial \mathcal{L}}{\partial z_q^{ST}}$$

梯度可以直接通过！

---

## 6. 完整损失函数

### 6.1 最终形式

结合所有损失项：

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{codebook} + \mathcal{L}_{commitment}$$

$$= \|x - D_\theta(z_q)\|^2 + \|z_e - \text{sg}(z_q)\|^2 + \beta \|\text{sg}(z_q) - z_e\|^2$$

### 6.2 梯度流分析

**对解码器参数 $\theta$ 的梯度**：
$$\frac{\partial \mathcal{L}_{total}}{\partial \theta} = \frac{\partial \mathcal{L}_{recon}}{\partial \theta} = 2(x - D_\theta(z_q)) \cdot \frac{\partial D_\theta(z_q)}{\partial \theta}$$

**对编码器参数 $\phi$ 的梯度**：
$$\frac{\partial \mathcal{L}_{total}}{\partial \phi} = \frac{\partial \mathcal{L}_{recon}}{\partial z_e} \cdot \frac{\partial z_e}{\partial \phi} + \beta \frac{\partial \mathcal{L}_{commitment}}{\partial z_e} \cdot \frac{\partial z_e}{\partial \phi}$$

由于直通估计器，$\frac{\partial \mathcal{L}_{recon}}{\partial z_e} = \frac{\partial \mathcal{L}_{recon}}{\partial z_q}$。

**对 codebook 向量 $e_k$ 的梯度**：
$$\frac{\partial \mathcal{L}_{total}}{\partial e_k} = \frac{\partial \mathcal{L}_{codebook}}{\partial e_k} = 2(z_e - e_k)$$

---

## 7. 算法流程

### 7.1 训练算法

**输入**：训练数据 $x$，codebook 大小 $K$，嵌入维度 $d$，commitment 权重 $\beta$

**初始化**：
1. 初始化编码器 $E_\phi$、解码器 $D_\theta$ 的参数
2. 初始化 codebook $\mathcal{E} = \{e_1, \ldots, e_K\}$（通常使用均匀分布或从编码器输出采样）

**训练循环**：
```
对于每个训练样本 x：
    1. 编码：z_e = E_φ(x)
    2. 量化：k = argmin_j ||z_e - e_j||²
            z_q = e_k
    3. 直通：z_q^ST = z_q + (z_e - sg(z_e))
    4. 解码：x̂ = D_θ(z_q^ST)
    5. 计算损失：
       L_recon = ||x - x̂||²
       L_codebook = ||z_e - sg(z_q)||²
       L_commitment = β ||sg(z_q) - z_e||²
       L_total = L_recon + L_codebook + L_commitment
    6. 反向传播并更新参数
```

### 7.2 生成过程

**从 codebook 采样生成新样本**：

```
1. 随机选择 codebook 索引：k ~ Uniform{1, ..., K}
2. 获取量化向量：z_q = e_k
3. 解码：x̂ = D_θ(z_q)
```

---

## 8. 关键公式总结

### 8.1 核心公式

**编码**：
$$z_e = E_\phi(x)$$

**量化**：
$$k = \arg\min_{j} \|z_e - e_j\|^2$$
$$z_q = e_k$$

**直通估计**：
$$z_q^{ST} = z_q + (z_e - \text{sg}(z_e))$$

**解码**：
$$x̂ = D_\theta(z_q^{ST})$$

### 8.2 损失函数

**重构损失**：
$$\mathcal{L}_{recon} = \|x - x̂\|^2$$

**Codebook 损失**：
$$\mathcal{L}_{codebook} = \|z_e - \text{sg}(z_q)\|^2$$

**Commitment 损失**：
$$\mathcal{L}_{commitment} = \beta \|\text{sg}(z_q) - z_e\|^2$$

**总损失**：
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{codebook} + \mathcal{L}_{commitment}$$

### 8.3 梯度更新

**解码器**：
$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}_{recon}}{\partial \theta}$$

**编码器**：
$$\phi \leftarrow \phi - \alpha \left(\frac{\partial \mathcal{L}_{recon}}{\partial \phi} + \beta \frac{\partial \mathcal{L}_{commitment}}{\partial \phi}\right)$$

**Codebook**：
$$e_k \leftarrow e_k - \alpha \frac{\partial \mathcal{L}_{codebook}}{\partial e_k}$$

其中 $\alpha$ 是学习率。

---

## 9. 与标准 VAE 的对比

### 9.1 潜在变量分布

**标准 VAE**：
- $q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \Sigma_\phi(x))$（连续分布）
- 使用重参数化技巧：$z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$

**VQ-VAE**：
- $q(z|x)$ 是离散分布（one-hot 在 codebook 上）
- $z = e_k$，其中 $k = \arg\min_j \|z_e - e_j\|^2$

### 9.2 正则化项

**标准 VAE**：
- KL 散度：$\text{KL}(q_\phi(z|x) \| p(z))$
- 鼓励潜在变量接近先验分布 $p(z) = \mathcal{N}(0, I)$

**VQ-VAE**：
- VQ 损失：$\|z_e - z_q\|^2$
- 鼓励编码器输出接近 codebook 向量

### 9.3 先验分布

**标准 VAE**：
- 先验：$p(z) = \mathcal{N}(0, I)$
- 生成时从先验采样：$z \sim \mathcal{N}(0, I)$

**VQ-VAE**：
- 先验：$p(z) = \text{Uniform}\{e_1, \ldots, e_K\}$（训练时）
- 生成时从 codebook 均匀采样：$z = e_k$，其中 $k \sim \text{Uniform}\{1, \ldots, K\}$
- 实际应用中，通常训练一个自回归模型（如 PixelCNN 或 Transformer）来建模 $p(z)$

---

## 10. 实现细节

### 10.1 Codebook 初始化

通常使用以下方法之一：

1. **均匀初始化**：
   $$e_j \sim \text{Uniform}(-\frac{1}{K}, \frac{1}{K})$$

2. **从编码器输出采样**：
   - 对一批训练样本编码得到 $\{z_e^{(i)}\}$
   - 使用 K-means 聚类初始化 codebook

### 10.2 Perplexity（困惑度）

用于监控 codebook 的使用情况：

$$\text{Perplexity} = \exp\left(-\sum_{j=1}^{K} p_j \log p_j\right)$$

其中 $p_j$ 是 codebook 向量 $e_j$ 被使用的频率。

- **高困惑度**：codebook 使用均匀，所有向量都被使用
- **低困惑度**：只有少数 codebook 向量被使用

理想情况下，希望 perplexity 接近 $K$（所有向量都被使用）。

### 10.3 指数移动平均（EMA）更新

一些实现使用 EMA 来更新 codebook，而不是直接梯度更新：

$$e_k \leftarrow \gamma e_k + (1 - \gamma) \bar{z}_e^{(k)}$$

其中 $\bar{z}_e^{(k)}$ 是映射到 $e_k$ 的所有编码器输出的平均值，$\gamma$ 是衰减率。

---

## 11. 参考文献

1. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017)**. Neural Discrete Representation Learning. *Advances in Neural Information Processing Systems*.

2. **Kingma, D. P., & Welling, M. (2014)**. Auto-Encoding Variational Bayes. *International Conference on Learning Representations*.

---

## 12. 总结

VQ-VAE 通过向量量化将连续潜在变量离散化，结合了 VAE 的表达能力和离散表示的优势。关键创新包括：

1. **向量量化层**：将连续编码映射到离散 codebook
2. **直通估计器**：解决离散操作的梯度问题
3. **双重损失**：codebook 损失和 commitment 损失确保编码器和 codebook 协同学习

这使得 VQ-VAE 能够学习有意义的离散表示，为后续的自回归生成模型（如 VQ-VAE-2、DALL-E）奠定了基础。


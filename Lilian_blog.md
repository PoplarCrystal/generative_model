# 从自编码器到 Beta-VAE

**原文链接**: https://lilianweng.github.io/posts/2018-08-12-vae/  
**作者**: Lilian Weng  
**翻译日期**: 2024

---

## 目录

1. [符号说明](#符号说明)
2. [自编码器](#自编码器)
3. [去噪自编码器](#去噪自编码器)
4. [稀疏自编码器](#稀疏自编码器)
5. [收缩自编码器](#收缩自编码器)
6. [变分自编码器（VAE）](#变分自编码器vae)
   - [损失函数：ELBO](#损失函数elbo)
   - [重参数化技巧](#重参数化技巧)
7. [β-VAE](#β-vae)
8. [VQ-VAE 和 VQ-VAE-2](#vq-vae-和-vq-vae-2)
9. [TD-VAE](#td-vae)
10. [参考文献](#参考文献)

---

## 符号说明

- $\mathcal{D}$：数据集，包含 $n$ 个数据样本
- $\mathbf{x}^{(i)}$：第 $i$ 个数据样本，是一个 $d$ 维向量
- $\mathbf{x}$：数据集中的一个样本
- $\mathbf{x}'$：$\mathbf{x}$ 的重构版本
- $\tilde{\mathbf{x}}$：$\mathbf{x}$ 的被扰动版本
- $\mathbf{z}$：在瓶颈层学习到的压缩编码
- $a_j^{(l)}$：第 $l$ 层中第 $j$ 个神经元的激活函数
- $g_{\phi}(.)$：由参数 $\phi$ 定义的编码函数
- $f_{\theta}(.)$：由参数 $\theta$ 定义的解码函数
- $q_{\phi}(\mathbf{z}|\mathbf{x})$：估计的后验概率函数，也称为概率编码器
- $p_{\theta}(\mathbf{x}|\mathbf{z})$：给定潜在编码生成真实数据样本的可能性，也称为概率解码器

---

## 自编码器

自编码器（Autoencoder）是一种神经网络，旨在以无监督的方式学习恒等函数，重构原始输入，同时在此过程中压缩数据，以发现更高效的表示。它由两个网络组成：

- **编码器网络**：将原始高维输入转换为低维潜在编码
- **解码器网络**：从编码中恢复数据

编码器网络实现了降维，与主成分分析（PCA）或矩阵分解（MF）类似。此外，自编码器明确优化了从编码重构数据的过程。

模型包含由参数 $\phi$ 定义的编码函数 $g(.)$ 和由参数 $\theta$ 定义的解码函数 $f(.)$。

对于输入 $\mathbf{x}$，在瓶颈层学习到的低维编码为：
$$\mathbf{z} = g_\phi(\mathbf{x})$$

重构的输入为：
$$\mathbf{x}' = f_\theta(g_\phi(\mathbf{x}))$$

参数 $(\theta, \phi)$ 共同学习，以输出与原始输入相同的重构数据样本，即 $\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$。

常用的损失函数包括均方误差（MSE）损失：

$$ L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2 $$

---

## 去噪自编码器

由于自编码器学习恒等函数，当网络参数多于数据点时，可能会出现过拟合的风险。为避免过拟合并提高鲁棒性，**去噪自编码器（Denoising Autoencoder, DAE）** 对基本自编码器进行了修改。

输入通过添加噪声或随机遮蔽部分值进行部分破坏，生成被扰动的输入 $\tilde{\mathbf{x}}$。然后，模型被训练以恢复原始输入（而非被扰动的输入）。

$$ \begin{aligned} \tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\ L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2 \end{aligned} $$

其中，$\mathcal{M}_\mathcal{D}$ 定义了从真实数据样本到被扰动样本的映射。

这种设计的动机在于，人类即使在视图部分被遮挡或破坏的情况下，也能轻松识别物体或场景。为了"修复"部分被破坏的输入，去噪自编码器必须发现并捕捉输入各维度之间的关系，以推断缺失部分。

---

## 稀疏自编码器

**稀疏自编码器（Sparse Autoencoder）** 在隐藏单元激活上施加"稀疏"约束，以避免过拟合并提高鲁棒性。它强制模型在同一时间仅有少量隐藏单元被激活，换句话说，一个隐藏神经元在大多数时间应处于非激活状态。

常见的激活函数包括 sigmoid、tanh、ReLU、Leaky ReLU 等。当值接近 1 时，神经元被激活；当值接近 0 时，神经元处于非激活状态。

---

## 收缩自编码器

类似于稀疏自编码器，**收缩自编码器（Contractive Autoencoder, CAE）** 鼓励学习到的表示保持在一个收缩空间，以提高鲁棒性。它在损失函数中添加一项，惩罚表示对输入过于敏感，从而提高对训练数据点周围小扰动的鲁棒性。

这种敏感性通过编码器激活对输入的雅可比矩阵的弗罗贝尼乌斯范数来衡量：

$$ L_\text{CAE}(\theta, \phi) = L_\text{AE}(\theta, \phi) + \lambda \sum_{i=1}^n \| \nabla_\mathbf{x} g_\phi(\mathbf{x}^{(i)}) \|_F^2 $$

其中，$\lambda$ 是正则化系数，$\| \cdot \|_F$ 表示弗罗贝尼乌斯范数。

---

## 变分自编码器（VAE）

**变分自编码器（Variational Autoencoder, VAE）** 的思想与上述所有自编码器模型有所不同，但深深植根于变分贝叶斯和图模型的方法中。

我们希望将输入映射到一个分布，而不是固定的向量。假设我们知道该分布的真实参数 $\theta^{*}$。为了生成一个看起来像真实数据点的样本，我们遵循以下步骤：

1. 首先，从先验分布 $p_{\theta^*}(\mathbf{z})$ 中采样一个 $\mathbf{z}^{(i)}$
2. 然后，从条件分布 $p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$ 中生成一个值 $\mathbf{x}^{(i)}$

### 损失函数：ELBO

估计的后验分布 $q_\phi(\mathbf{z}|\mathbf{x})$ 应该非常接近真实的后验分布 $p_\theta(\mathbf{z}|\mathbf{x})$。我们可以使用 Kullback-Leibler（KL）散度来量化这两个分布之间的距离。在我们的情况下，我们希望最小化 $D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}) )$，以使估计的后验分布尽可能接近真实的后验分布。

然而，直接计算 KL 散度是很困难的，因为我们不知道真实的后验分布 $p_\theta(\mathbf{z}|\mathbf{x})$。幸运的是，我们可以通过最大化证据下界（Evidence Lower BOund, ELBO）来间接最小化 KL 散度：

$$ \begin{aligned} D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) &= \int q_\phi(\mathbf{z}|\mathbf{x}) \log\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} d\mathbf{z} \\ &= \int q_\phi(\mathbf{z}|\mathbf{x}) \log\frac{q_\phi(\mathbf{z}|\mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} \\ &= \int q_\phi(\mathbf{z}|\mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} \\ &= \log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z}|\mathbf{x})\log\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} \\ &= \log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z}|\mathbf{x})\log\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{x}|\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} \\ &= \log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}[\log q_\phi(\mathbf{z}|\mathbf{x}) - \log p_\theta(\mathbf{x}|\mathbf{z}) - \log p_\theta(\mathbf{z})] \\ &= \log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}\log p_\theta(\mathbf{x}|\mathbf{z}) \end{aligned} $$

重新排列后，我们有：

$$ \log p_\theta(\mathbf{x}) = D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) + \mathcal{L}(\theta, \phi; \mathbf{x}) $$

其中：

$$ \mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}\log p_\theta(\mathbf{x}|\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z})) $$

$\mathcal{L}(\theta, \phi; \mathbf{x})$ 被称为**证据下界（ELBO）**。由于 KL 散度总是非负的，我们有：

$$ \log p_\theta(\mathbf{x}) \geq \mathcal{L}(\theta, \phi; \mathbf{x}) $$

因此，最大化 ELBO 等价于最小化 KL 散度。

ELBO 由两项组成：
1. **重构项**：$\mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}\log p_\theta(\mathbf{x}|\mathbf{z})$ - 鼓励模型从潜在编码 $\mathbf{z}$ 重构原始输入 $\mathbf{x}$
2. **正则化项**：$-D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}))$ - 鼓励估计的后验分布 $q_\phi(\mathbf{z}|\mathbf{x})$ 接近先验分布 $p_\theta(\mathbf{z})$

### 重参数化技巧

损失函数中的期望项涉及从 $q_\phi(\mathbf{z}|\mathbf{x})$ 中采样 $\mathbf{z}$。采样是一个随机过程，因此我们无法对其进行梯度反向传播。为了使其可训练，引入了**重参数化技巧（Reparameterization Trick）**。

通常可以将随机变量 $\mathbf{z}$ 表示为一个确定性变量：
$$\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$$

其中 $\boldsymbol{\epsilon}$ 是一个辅助的独立随机变量，变换函数 $\mathcal{T}_\phi$ 将 $\boldsymbol{\epsilon}$ 和 $\mathbf{x}$ 映射到 $\mathbf{z}$。

对于 VAE，我们通常假设 $q_\phi(\mathbf{z}|\mathbf{x})$ 是一个多元高斯分布：
$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x})\mathbf{I})$$

使用重参数化技巧：
$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\odot$ 表示逐元素相乘。

现在，梯度可以通过 $\boldsymbol{\mu}_\phi(\mathbf{x})$ 和 $\boldsymbol{\sigma}_\phi(\mathbf{x})$ 反向传播，而不是通过随机采样过程。

---

## β-VAE

**β-VAE** 是 VAE 的一个变体，通过引入超参数 $\beta$ 来平衡重构误差和正则化项：

$$ \mathcal{L}_\beta(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}\log p_\theta(\mathbf{x}|\mathbf{z}) - \beta \cdot D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z})) $$

当 $\beta = 1$ 时，这是标准的 VAE。当 $\beta > 1$ 时，模型更强调潜在空间的独立性，从而鼓励学习更具解耦性的表示。$\beta$ 值越大，模型就越倾向于学习更独立的潜在因子，这有助于学习可解释的表示。

---

## VQ-VAE 和 VQ-VAE-2

**向量量化 VAE（Vector Quantised Variational AutoEncoder, VQ-VAE）** 引入了一种离散的潜在表示方法，而不是连续的高斯分布。

VQ-VAE 使用一个离散的潜在嵌入空间 $e \in \mathbb{R}^{K \times D}$，其中 $K$ 是嵌入空间的大小，$D$ 是每个潜在嵌入向量的维度。

模型的架构如下：

1. **编码器**：$E$ 输出连续的向量 $z_e(\mathbf{x})$
2. **向量量化**：使用最近邻查找将 $z_e$ 量化为离散的嵌入向量 $z_q(\mathbf{x})$：
   $$z_q(\mathbf{x}) = e_k, \quad \text{其中 } k = \arg\min_j \|z_e(\mathbf{x}) - e_j\|_2$$
3. **解码器**：$G$ 从量化后的向量 $z_q$ 重构数据

损失函数包括三个部分：

$$L = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_\text{重构损失} + \underbrace{\|sg[z_e(\mathbf{x})] - e_k\|_2^2}_\text{VQ损失} + \underbrace{\beta\|z_e(\mathbf{x}) - sg[e_k]\|_2^2}_\text{承诺损失}$$

其中 $sg[\cdot]$ 表示停止梯度操作（stop-gradient）。VQ 损失更新嵌入向量 $e_k$，承诺损失（commitment loss）鼓励编码器输出接近所选嵌入向量。

**VQ-VAE-2** 是对 VQ-VAE 的改进，使用分层向量量化来生成更高分辨率的图像。它包含两个层次的潜在空间：
- 顶层：捕捉全局信息
- 底层：捕捉局部细节

这种分层方法使得模型能够生成更高质量的图像，并保持长期依赖关系。

---

## TD-VAE

**时间差分 VAE（Temporal Difference VAE, TD-VAE）** 是一种用于建模序列数据的变分自编码器。它旨在学习状态的抽象表示，并能够进行时间抽象推理。

TD-VAE 的关键思想是：
1. 学习状态表示 $b_t$，捕捉时间 $t$ 的状态信息
2. 使用信念状态（belief state）$q(b_t | \mathbf{x}_{\leq t})$ 来推断当前状态
3. 通过跳跃式预测（jumpy prediction）来学习时间抽象

模型能够：
- 从部分观测序列推断状态
- 预测未来状态，即使跨越多个时间步
- 进行反事实推理（what-if scenarios）

TD-VAE 的损失函数结合了变分下界和跳跃预测项，使模型能够学习有意义的抽象表示，并在时间上进行推理。

---

## 参考文献

1. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.

2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. *International conference on machine learning*.

3. Higgins, I., et al. (2017). beta-vae: Learning basic visual concepts with a constrained variational framework. *International Conference on Learning Representations*.

4. van den Oord, A., Vinyals, O., & kavukcuoglu, k. (2017). Neural discrete representation learning. *Advances in neural information processing systems*.

5. Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating diverse high-fidelity images with vq-vae-2. *Advances in neural information processing systems*.

6. Gregor, K., et al. (2019). Temporal difference variational auto-encoder. *International Conference on Learning Representations*.

7. Vincent, P., et al. (2010). Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *Journal of machine learning research*.

8. Rifai, S., et al. (2011). Higher order contractive auto-encoder. *Joint European conference on machine learning and knowledge discovery in databases*.

---

**翻译说明**：本文档是对 Lilian Weng 的博客文章《From Autoencoder to Beta-VAE》的中文翻译。原文链接：https://lilianweng.github.io/posts/2018-08-12-vae/
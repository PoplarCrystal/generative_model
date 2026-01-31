# GAN系列篇03. CGAN：从"无条件生成"到"条件生成"（理论篇）

## 1. GAN的问题与CGAN的引入

### 1.1 GAN的局限性

生成对抗网络（GAN）是一种强大的生成模型，它通过对抗训练让生成分布 $p_g(x)$ 逼近真实数据分布 $p_{data}(x)$，从而能够生成逼真的样本。标准GAN从随机噪声 $z \sim p_z(z)$ 生成样本，通过生成器 $G(z)$ 学习到的分布 $p_g(x)$ 来生成新样本。

然而，传统的GAN在处理一些需要特定条件控制的生成任务时显得力不从心。具体来说：

**无法控制生成内容**：标准GAN从随机噪声 $z \sim p_z(z)$ 生成样本，生成的内容是完全随机的，无法指定生成特定类别的样本。例如，在MNIST数据集上，我们无法控制生成器生成数字"7"的图像，只能随机生成任意数字。

### 1.2 CGAN的解决方案

在图像生成任务中，我们可能希望根据特定标签（如数字"7"）来生成相应的图像，而不是随机生成一个图像。

为了增强GAN的灵活性和可控性，提出了**条件生成对抗网络（Conditional Generative Adversarial Network, CGAN）**。CGAN允许我们在生成过程中加入额外的信息（条件），从而实现更加精确和可控的数据生成。

**核心思想**：在GAN的基础上，引入条件变量 $c$，使得生成过程依赖于条件信息。通过指定条件变量 $c$（如类别标签、文本描述等）来指导模型生成特定类别的样本。

为了更好地理解GAN和CGAN的区别，我们先回顾一下标准GAN的架构：

**标准GAN**：
- 生成器：$G(z) \rightarrow x$，输入随机噪声 $z$，输出样本 $x$
- 判别器：$D(x) \rightarrow [0,1]$，输入样本 $x$，输出真实概率

<div align="center">
<img src="pic/dl/gan/3.1.jpeg" width="70%">
</div>

<small>**图1** 标准GAN的框架图：生成器从噪声 $z$ 生成样本 $x$，判别器判断样本 $x$ 是否真实</small>

## 2. CGAN的核心问题

### 2.1 问题1：条件c如何加入模型？

在CGAN中，条件变量 $c$ 需要同时加入到生成器和判别器中，形成条件对抗。

**生成器 $G(z \vert c)$**：
- 输入：噪声 $z \sim p_z(z)$ + 条件 $c$
- 输出：以 $c$ 为条件的假样本 $x' = G(z \vert c)$
- 目标：生成符合条件 $c$ 的逼真样本

**判别器 $D(x \vert c)$**：
- 输入：样本 $x$ + 条件 $c$
- 输出：该样本在给定条件 $c$ 下为真实的概率 $D(x \vert c) \in [0,1]$
- 目标：判断 $(x, c)$ 是否匹配（即 $x$ 是否属于类别 $c$）

<div align="center">
<img src="pic/dl/gan/3.2.jpeg" width="70%">
</div>

<small>**图2** CGAN的框架图：生成器从噪声 $z$ 和条件 $c$ 生成样本 $x$，判别器判断样本 $x$ 在给定条件 $c$ 下是否真实</small>

✅ **关键**：**G 和 D 都"知道"当前的条件 $c$**，形成条件对抗。这与标准GAN的区别在于，判别器不仅要判断样本是否真实，还要判断样本是否与条件匹配。

### 2.2 问题2：条件c如何编码？

条件变量 $c$ 通常是离散标签（如 MNIST 的 0~9），需要转换为数值向量才能输入神经网络。常见的编码方式有：

**方式1：One-Hot编码**（最常用）
- 方法：将类别索引转换为独热向量
- 例如：类别3 → $[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$（MNIST的10个类别）
- 优点：简单直观，不需要额外参数
- 缺点：维度等于类别数，当类别数很大时维度很高；编码是固定的，无法学习

**方式2：Embedding编码**
- 方法：使用可学习的嵌入层将类别索引映射为低维向量
- 例如：类别3 → 可学习的N维向量
- 优点：维度可自定义，通常比one-hot更紧凑；可以学习到类别之间的语义关系
- 缺点：需要额外的可学习参数

**实现示例（MNIST, 10类，One-Hot编码）**：

```python
# 生成器输入
z = torch.randn(batch_size, latent_dim)          # [B, 100]
c = F.one_hot(labels, num_classes=10).float()    # [B, 10]
input_g = torch.cat([z, c], dim=1)               # [B, 110]
x_fake = G(input_g)

# 判别器输入
x_flat = x.view(batch_size, -1)                   # [B, 784]
input_d = torch.cat([x_flat, c], dim=1)          # [B, 794]
output = D(input_d)                               # [B, 1]
```

## 3. CGAN的损失函数推导

### 3.1 从条件概率建模出发

CGAN的核心是将无条件生成问题转化为条件生成问题。现在的问题变成：给定条件 $c$，判断 $x$ 是来自真实条件分布 $p_{data}(x|c)$ 还是生成分布 $p_g(x|c)$。

### 3.2 判别器的目标

判别器要最大化以下对数似然：

$$L_D = \mathbb{E}_{x \sim p_{data}(x|c)} \left[ \log D(x \vert c) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left(1 - D(G(z \vert c) \vert c) \right) \right]$$

其中：
- **第一项**：真实样本对 $(x, c)$ 应被判为真（输出 $\approx 1$），这里 $(x, c) \sim p_{data}(x,c)$ 表示从数据集中采样 $(x, c)$ 对
- **第二项**：生成样本对 $(G(z \vert c), c)$ 应被判为假（输出 $\approx 0$），这里 $(x, c) \sim p_{data}(x,c)$（从数据集中采样条件 $c$），$z \sim p_z(z)$（从先验分布采样噪声），$G(z \vert c) \sim p_g(x|c)$（生成器输出）

📌 **注意**：在训练时，我们从数据集中采样 $(x, c)$ 对，条件 $c$ 自然来自数据分布。判别器不仅要判断样本是否真实，还要判断样本是否与条件匹配。

### 3.3 生成器的目标

生成器希望欺骗判别器，即让 $D(G(z \vert c) \vert c)$ 尽可能大。

**原始目标（理论形式）**：
$$\min_G \mathbb{E}_{z \sim p_z(z)} \left[ \log \left(1 - D(G(z \vert c) \vert c) \right) \right]$$

**实践中采用非饱和版本**（避免梯度消失）：
$$\max_G \mathbb{E}_{z \sim p_z(z)} \left[ \log D(G(z \vert c) \vert c) \right]$$

这与标准GAN中生成器的非饱和损失形式一致，在训练早期能提供更强的梯度信号。

### 3.4 完整的Min-Max目标函数

将两者结合，得到CGAN的标准损失函数：

$$
\min_G \max_D L(D, G) = \mathbb{E}_{x \sim p_{data}(x|c)} \left[ \log D(x \vert c) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left(1 - D(G(z \vert c) \vert c) \right) \right]
 \tag{1} $$

🔸 **对比标准GAN**：只是在 $D$ 和 $G$ 的输入中**显式加入了条件 $c$**，并将所有分布条件化！

**标准GAN的目标函数**：
$$\min_G \max_D L(D, G) = \mathbb{E}_{x \sim p_{data}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left(1 - D(G(z)) \right) \right]$$

**CGAN的目标函数**：
$$\min_G \max_D L(D, G) = \mathbb{E}_{x \sim p_{data}(x|c)} \left[ \log D(x \vert c) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left(1 - D(G(z \vert c) \vert c) \right) \right]$$

可以看出，CGAN只是在标准GAN的基础上，将所有分布都条件化于 $c$：$p_{data}(x) \rightarrow p_{data}(x|c)$，$p_g(x) \rightarrow p_g(x|c)$。

## 4. 理论解释：与条件JS散度的关系

### 4.1 最优判别器

类似标准GAN，可以证明当判别器最优时：

$$D^*(x \vert c) = \frac{p_{data}(x|c)}{p_{data}(x|c) + p_g(x|c)}$$

这个结果与标准GAN类似，只是将无条件分布替换为条件分布。判别器输出的是：给定条件 $c$，样本 $x$ 来自真实条件分布 $p_{data}(x|c)$ 的概率。

### 4.2 条件JS散度

将最优判别器代入目标函数后，可得：

$$L(D^*, G) = 2 \mathrm{JS}\left( p_{data}(x|c) \| p_g(x|c) \right) -2\log 2 $$

📌 **结论**：CGAN的训练等价于最小化条件分布 $p_g(x|c)$ 与真实条件分布 $p_{data}(x|c)$ 之间的**期望JS散度**。

当对所有条件 $c$ 都有 $p_g(x|c) = p_{data}(x|c)$ 时，达到全局最优。

## 5. Conditional的作用与本质

### 5.1 Conditional的作用

在CGAN中，条件变量 $c$ 起到以下关键作用：

1. **控制生成内容**：
   - 通过指定不同的条件 $c$，可以生成不同类别的样本
   - 例如：在MNIST上，$c$ 可以是数字标签（0-9），生成对应数字的图像

2. **增强判别能力**：
   - 判别器不仅要判断样本是否真实，还要判断样本是否与条件匹配
   - 这有助于生成器学习生成与条件一致的样本

3. **提高训练稳定性**：
   - 条件信息为生成器和判别器提供了额外的监督信号
   - 有助于避免mode collapse问题

### 5.2 Conditional的本质：条件对抗

**核心思想**：CGAN中条件 $c$ 的本质作用是**将无条件对抗转化为条件对抗**。

#### 5.2.1 标准GAN中的对抗

在标准GAN中：
- 生成器：$G(z) \rightarrow x$，从噪声生成样本
- 判别器：$D(x) \rightarrow [0,1]$，判断样本是否真实
- 对抗：生成器试图生成逼真样本，判别器试图区分真假

#### 5.2.2 CGAN中的条件对抗

在CGAN中：
- 生成器：$G(z \vert c) \rightarrow x$，从噪声和条件生成样本
- 判别器：$D(x \vert c) \rightarrow [0,1]$，判断样本在给定条件下是否真实
- 条件对抗：生成器试图生成与条件匹配的逼真样本，判别器试图判断样本是否真实且与条件匹配

#### 5.2.3 数学表达

**标准GAN**：
$$\min_G \max_D L(D, G) = \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]$$

**CGAN**：
$$\min_G \max_D L(D, G) = \mathbb{E}_{x \sim p_{data}(x|c)} [\log D(x \vert c)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z \vert c) \vert c))]$$

条件 $c$ 将无条件分布 $p(x)$ 分解为条件分布 $p(x|c)$，使得生成过程可控。

---

## 📚 参考

- Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*.
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014.

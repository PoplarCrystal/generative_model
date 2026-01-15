# 深度学习AE篇05. VAE-CG: 从"条件生成"到"引导生成"

## 1. CVAE的问题与VAE-CG的引入

### 1.1 CVAE的局限性

条件变分自编码器（CVAE）通过引入条件变量 $c$ 实现了条件生成，能够根据特定条件（如类别标签）生成相应的样本。然而，CVAE在处理某些生成任务时仍存在局限性：

1. **生成质量与条件控制的权衡**：
   - CVAE需要在训练时同时学习条件信息和解码能力
   - 在条件约束下，生成质量可能受到限制
   - 难以在生成质量和条件遵循之间找到最佳平衡

2. **条件信息的利用不充分**：
   - CVAE将条件信息直接注入编码器和解码器
   - 对于某些任务，可能需要更灵活的方式来利用条件信息
   - 条件信息在生成过程中的作用相对固定

3. **生成过程的控制能力有限**：
   - CVAE的生成过程是确定性的（给定条件 $c$ 和采样 $z$）
   - 无法在生成过程中动态调整条件的影响强度
   - 难以实现"渐进式"或"可调节"的条件生成

**示例场景**：
- 在图像生成任务中，我们希望生成质量高且符合条件的样本
- 在某些情况下，可能需要"强化"条件的影响（更强的条件遵循）
- 在其他情况下，可能需要"弱化"条件的影响（更自由的生成）

### 1.2 VAE-CG的解决方案

为了增强CVAE的条件控制能力，提出了**基于引导生成的VAE变种（VAE with Guided Generation）**，主要包括两种方式：

1. **Classifier Guidance (CG)**：使用额外的分类器在潜在空间引导生成
2. **Classifier-Free Guidance (CFG)**：通过训练条件和无条件模型，在生成时组合两者

**核心思想**：
- **引导生成**：在生成过程中，通过某种"引导信号"来调整生成方向
- **动态控制**：可以通过引导强度（guidance scale）动态调整条件的影响
- **质量提升**：引导机制可以帮助生成更高质量、更符合条件的样本

这些方法最初在扩散模型（如LDM）中取得了巨大成功，随后被应用到VAE框架中。

## 2. Classifier Guidance (CG): 分类器引导生成

### 2.1 CG的基本原理

**Classifier Guidance** 的核心思想是：使用一个**额外的分类器**在潜在空间中对生成过程进行引导。

**关键组件**：
1. **VAE模型**：负责学习数据的潜在表示和生成
2. **潜在空间分类器**：在潜在变量 $z$ 上预测类别标签 $y$

**生成过程**：
1. 从先验分布采样初始潜在变量 $z$
2. 使用分类器的梯度来"引导" $z$ 向目标类别方向移动
3. 通过迭代更新，得到符合目标类别的 $z$
4. 将引导后的 $z$ 输入解码器生成样本

### 2.2 CG的数学原理

**生成目标**：我们希望生成符合条件 $y$ 的样本。

**条件生成概率**：
$$p(x|y) = \int p(x|z) p(z|y) dz$$

其中，$p(z|y)$ 是给定类别 $y$ 的潜在变量分布。

**关键洞察**：使用贝叶斯公式，我们可以将 $p(z|y)$ 表示为：
$$p(z|y) = \frac{p(y|z) p(z)}{p(y)} \propto p(y|z) p(z)$$

因此，给定类别 $y$ 的潜在变量分布与分类器 $p(y|z)$ 和先验 $p(z)$ 的乘积成正比。

**引导目标**：我们希望从 $p(z|y)$ 中采样。通过对 $\log p(z|y)$ 进行梯度上升，我们可以将 $z$ 向高概率区域移动：
$$\nabla_z \log p(z|y) = \nabla_z [\log p(y|z) + \log p(z)] = \nabla_z \log p(y|z) + \nabla_z \log p(z)$$

由于 $p(z) = \mathcal{N}(0, I)$ 是标准正态分布，$\nabla_z \log p(z) = -z$（取决于具体形式）。

**CG更新规则**：使用梯度引导更新 $z$：
$$z_{t+1} = z_t + \lambda \cdot \nabla_z \log p(y|z_t) + \text{noise}$$

其中，$\lambda$ 是引导强度（guidance scale），通常是一个小的步长。

### 2.3 CG的训练过程

**两阶段训练**：

**阶段1：训练VAE**
- 使用标准VAE损失函数训练编码器和解码器
- 可以选择是否使用条件（类似CVAE）

**阶段2：训练分类器**
- 在潜在空间训练分类器 $p_\theta(y|z)$
- 使用VAE编码器得到的潜在变量 $z$ 和对应的标签 $y$ 训练分类器
- 分类器损失：$\mathcal{L}_{classifier} = -\mathbb{E}_{(x,y)}[\log p_\theta(y|z)]$，其中 $z \sim q_\phi(z|x)$

### 2.4 CG的采样过程

**Langevin动力学风格采样**：

1. **初始化**：从先验分布采样 $z_0 \sim \mathcal{N}(0, I)$

2. **迭代更新**（$t = 0, 1, \ldots, T-1$）：
   ```python
   # 计算分类器输出的log概率
   logits = classifier(z_t)
   log_probs = log_softmax(logits)
   target_log_prob = log_probs[y]  # 目标类别的log概率
   
   # 计算梯度
   grad = ∇_z target_log_prob
   
   # 更新z（梯度上升 + 噪声）
   z_{t+1} = z_t + guidance_scale * step_size * grad + noise
   ```

3. **生成样本**：使用最终得到的 $z_T$ 输入解码器：$x = \text{decoder}(z_T)$

**关键参数**：
- **guidance_scale**：引导强度，控制条件影响的大小
- **step_size**：更新步长，通常很小（如0.01）
- **noise**：保持探索性的噪声项

### 2.5 CG的优缺点

**优点**：
1. **灵活的条件控制**：可以通过调整引导强度动态控制条件影响
2. **高质量生成**：引导机制有助于生成更符合条件的样本
3. **简单直观**：概念清晰，易于理解和实现

**缺点**：
1. **需要额外分类器**：需要训练和维护额外的分类器模型
2. **计算开销**：采样时需要多次迭代计算梯度，计算成本较高
3. **分类器质量依赖**：生成质量依赖于分类器的质量

## 3. Classifier-Free Guidance (CFG): 无分类器引导生成

### 3.1 CFG的基本原理

**Classifier-Free Guidance** 的核心思想是：**不需要额外的分类器**，而是通过训练模型同时支持条件和非条件生成，然后在生成时组合两者。

**关键创新**：
1. **统一模型**：同一个模型既支持条件生成 $p(x|y)$，也支持无条件生成 $p(x)$
2. **训练策略**：训练时随机丢弃条件（以一定概率将条件设为None）
3. **引导组合**：生成时同时计算条件和无条件输出，通过引导公式组合

### 3.2 CFG的训练过程

**训练策略：随机条件丢弃**

在训练时，以概率 $p_{uncond}$ 随机丢弃条件：
- 以概率 $1 - p_{uncond}$：使用条件 $y$ 训练条件模型 $p_\theta(x|y)$
- 以概率 $p_{uncond}$：不使用条件，训练无条件模型 $p_\theta(x)$

**损失函数**：
$$\mathcal{L}_{CFG} = \mathbb{E}_{(x,y) \sim p_{data}} \left[ (1-\epsilon) \cdot \mathcal{L}(x, y) + \epsilon \cdot \mathcal{L}(x, \emptyset) \right]$$

其中，$\epsilon \sim \text{Bernoulli}(p_{uncond})$，$\mathcal{L}(x, y)$ 是条件VAE的损失，$\mathcal{L}(x, \emptyset)$ 是无条件VAE的损失。

**关键点**：
- 模型需要能够处理条件为 `None` 的情况
- 编码器和解码器需要在条件存在和缺失时都能正常工作
- 通常设置 $p_{uncond} = 0.1$ 到 $0.2$

### 3.3 CFG的采样过程

**引导组合公式**：

在生成时，我们同时计算：
- **条件输出**：$\hat{x}_{cond} = \text{decoder}(z, y)$
- **无条件输出**：$\hat{x}_{uncond} = \text{decoder}(z, \emptyset)$

然后使用引导公式组合：
$$\hat{x}_{guided} = \hat{x}_{uncond} + w \cdot (\hat{x}_{cond} - \hat{x}_{uncond})$$

其中，$w$ 是引导强度（guidance scale），通常 $w > 1$。

**公式解释**：
- 当 $w = 0$ 时：$\hat{x}_{guided} = \hat{x}_{uncond}$，完全无条件生成
- 当 $w = 1$ 时：$\hat{x}_{guided} = \hat{x}_{cond}$，完全条件生成
- 当 $w > 1$ 时：$\hat{x}_{guided}$ 被"推离"无条件输出，向条件输出方向移动，从而强化条件的影响

**等价形式**：
$$\hat{x}_{guided} = (1 + w) \cdot \hat{x}_{cond} - w \cdot \hat{x}_{uncond}$$

### 3.4 CFG的数学原理

**理论推导**：

在扩散模型中，CFG的原理可以更清楚地解释。在VAE框架中，我们可以类似地理解：

**条件生成**：
$$p_\theta(x|y) = \int p_\theta(x|z, y) p(z) dz$$

**无条件生成**：
$$p_\theta(x) = \int p_\theta(x|z) p(z) dz$$

**引导生成**（启发式组合）：
$$p_\theta(x|y)_{guided} \propto p_\theta(x|y)^w \cdot p_\theta(x)^{1-w}$$

在log空间中：
$$\log p_\theta(x|y)_{guided} = w \cdot \log p_\theta(x|y) + (1-w) \cdot \log p_\theta(x)$$

这可以近似为：
$$\log p_\theta(x|y)_{guided} \approx \log p_\theta(x) + w \cdot (\log p_\theta(x|y) - \log p_\theta(x))$$

在解码器输出层面，这对应：
$$\hat{x}_{guided} = \hat{x}_{uncond} + w \cdot (\hat{x}_{cond} - \hat{x}_{uncond})$$

### 3.5 CFG的优缺点

**优点**：
1. **无需额外模型**：不需要训练额外的分类器，所有功能集成在一个模型中
2. **计算效率高**：采样时只需前向传播两次（条件和无条件），比CG的迭代更新更快
3. **更好的训练一致性**：条件和无条件模型在同一训练过程中学习，更加一致
4. **灵活性强**：可以轻松调整引导强度，实现不同强度的条件控制

**缺点**：
1. **训练复杂度**：需要同时训练条件和无条件模型，训练可能更复杂
2. **模型容量需求**：模型需要同时学习两种模式，可能需要更大的模型容量
3. **条件丢弃概率选择**：需要仔细选择 $p_{uncond}$ 以获得最佳性能

## 4. CG vs CFG: 对比分析

### 4.1 架构对比

| 特性 | Classifier Guidance (CG) | Classifier-Free Guidance (CFG) |
|------|-------------------------|-------------------------------|
| **额外模型** | 需要分类器 | 不需要额外模型 |
| **模型数量** | VAE + 分类器 | 单个统一模型 |
| **训练复杂度** | 两阶段训练 | 单阶段训练（需要条件丢弃） |
| **条件处理** | 分类器在潜在空间操作 | 条件直接输入VAE |

### 4.2 生成过程对比

| 特性 | CG | CFG |
|------|----|----|
| **采样方式** | 迭代梯度更新（Langevin风格） | 单次前向传播组合 |
| **计算成本** | 高（多次迭代） | 低（两次前向传播） |
| **引导机制** | 梯度引导 | 输出组合 |
| **参数控制** | 引导强度、步长、迭代次数 | 引导强度 |

### 4.3 性能对比

| 特性 | CG | CFG |
|------|----|----|
| **生成质量** | 依赖于分类器质量 | 通常更稳定 |
| **条件遵循** | 可调节，依赖于引导强度 | 可调节，更灵活 |
| **训练稳定性** | 分类器训练可能不稳定 | 通常更稳定 |
| **推理速度** | 较慢（迭代更新） | 较快（单次组合） |

### 4.4 适用场景

**Classifier Guidance 适用于**：
- 需要额外分类器的场景（如已有预训练分类器）
- 分类任务和生成任务可以分离的场景
- 对生成速度要求不高的场景

**Classifier-Free Guidance 适用于**：
- 希望使用单一统一模型的场景
- 需要快速推理的场景
- 训练和生成需要一致性的场景
- 当前的主流选择（如Stable Diffusion等）

## 5. 实现细节

### 5.1 VAE-CG架构

**VAE_CG 模型结构**：

```
VAE_CG
├── Encoder: 编码器（可选条件输入）
├── Decoder: 解码器（可选条件输入）
└── LatentClassifier: 潜在空间分类器
    ├── Linear(latent_dim → hidden_dim)
    └── Linear(hidden_dim → n_classes)
```

**训练过程**：
1. **VAE训练**：使用标准CVAE损失函数
2. **分类器训练**：在潜在变量 $z$ 上训练分类器

**损失函数**：
$$\mathcal{L}_{VAE\_CG} = \mathcal{L}_{VAE} + \alpha \cdot \mathcal{L}_{classifier}$$

其中：
- $\mathcal{L}_{VAE}$ 是VAE的重构损失和KL散度
- $\mathcal{L}_{classifier}$ 是分类器的交叉熵损失
- $\alpha$ 是平衡权重

### 5.2 VAE-CFG架构

**VAE_CFG 模型结构**：

```
VAE_CFG
├── Encoder: 编码器（支持条件为None）
└── Decoder: 解码器（支持条件为None）
```

**关键实现细节**：
1. **条件处理**：当条件为 `None` 时，使用零向量或跳过条件拼接
2. **训练时的随机丢弃**：以概率 $p_{uncond}$ 将条件设为 `None`
3. **生成时的组合**：同时计算条件和无条件输出，然后组合

**损失函数**：
$$\mathcal{L}_{VAE\_CFG} = \mathbb{E}_{(x,y)} \left[ \mathcal{L}_{VAE}(x, y') \right]$$

其中 $y' = y$ 或 $y' = \emptyset$（根据随机丢弃概率）。

### 5.3 采样算法

**CG采样算法**：

```python
def sample_with_cg(classifier, decoder, y, guidance_scale=1.0, num_steps=10):
    # 初始化
    z = sample_prior()  # 从 N(0,I) 采样
    
    for t in range(num_steps):
        # 计算分类器输出
        logits = classifier(z)
        log_probs = log_softmax(logits)
        target_log_prob = log_probs[y]
        
        # 计算梯度
        grad = ∇_z target_log_prob
        
        # 更新z
        z = z + guidance_scale * step_size * grad + noise
        
    # 生成样本
    x = decoder(z, y)
    return x
```

**CFG采样算法**：

```python
def sample_with_cfg(decoder, y, guidance_scale=1.0):
    # 从先验采样z
    z = sample_prior()
    
    # 计算条件输出
    x_cond = decoder(z, y)
    
    # 计算无条件输出
    x_uncond = decoder(z, None)
    
    # 引导组合
    x_guided = x_uncond + guidance_scale * (x_cond - x_uncond)
    
    return x_guided
```

## 6. 引导强度的作用

### 6.1 引导强度的影响

**Classifier Guidance**：
- **guidance_scale = 0**：无引导，等同于无条件生成
- **guidance_scale = 1**：正常引导强度
- **guidance_scale > 1**：强引导，更严格遵循条件，但可能过度约束
- **guidance_scale < 1**：弱引导，条件影响较小

**Classifier-Free Guidance**：
- **guidance_scale = 0**：完全无条件生成（$w = 0$）
- **guidance_scale = 1**：完全条件生成（$w = 1$）
- **guidance_scale > 1**：超条件生成（$w > 1$），强化条件影响
- 通常使用 $w = 2.0$ 到 $w = 7.5$ 之间的值

### 6.2 权衡曲线

引导强度的影响存在一个权衡：

- **低引导强度**：
  - ✓ 生成多样性高
  - ✓ 生成质量自然
  - ✗ 条件遵循较弱

- **高引导强度**：
  - ✓ 条件遵循强
  - ✓ 生成更精确符合条件
  - ✗ 可能过度约束
  - ✗ 生成多样性降低
  - ✗ 可能出现模式崩塌

### 6.3 实际应用中的选择

在实际应用中，引导强度的选择通常需要：
1. **实验调优**：通过实验找到最佳引导强度
2. **任务相关**：不同任务可能需要不同的引导强度
3. **数据相关**：不同数据集可能有不同的最优值
4. **动态调整**：某些场景可能需要动态调整引导强度

## 7. VAE-CG的应用与展望

### 7.1 应用场景

VAE-CG和VAE-CFG可以应用于各种条件生成任务：

1. **图像生成**：
   - 根据类别标签生成图像
   - 根据文本描述生成图像（需要文本编码器）
   - 风格迁移和控制

2. **数据增强**：
   - 生成特定类别的样本用于平衡数据集
   - 生成稀有类别的样本

3. **可控生成**：
   - 通过调整引导强度控制生成质量
   - 渐进式生成（从低引导到高引导）

### 7.2 与扩散模型的联系

VAE-CG和CFG的概念最初在扩散模型（如Stable Diffusion）中取得了巨大成功：

- **扩散模型的CFG**：在去噪过程中使用CFG，显著提升了生成质量和条件遵循
- **VAE框架的扩展**：将CFG引入VAE框架，提供了另一种实现引导生成的方式
- **统一的视角**：两种框架都体现了"引导生成"这一核心思想

### 7.3 未来方向

1. **自适应引导**：根据样本自动调整引导强度
2. **多条件引导**：同时处理多个条件（如类别+风格+属性）
3. **引导机制的理论分析**：更深入理解引导机制的理论基础
4. **效率优化**：提高CG的采样效率，减少迭代次数

---

## 8. 总结

VAE-CG通过引入**引导生成**机制，实现了从"条件生成"到"引导生成"的转变：

1. **Classifier Guidance (CG)**：
   - 使用额外分类器在潜在空间引导生成
   - 通过梯度更新迭代优化潜在变量
   - 灵活但计算成本较高

2. **Classifier-Free Guidance (CFG)**：
   - 统一模型同时支持条件和无条件生成
   - 通过输出组合实现引导
   - 高效且训练一致性好

3. **核心优势**：
   - 动态控制条件影响强度
   - 提升生成质量和条件遵循
   - 灵活的生成控制能力

4. **应用价值**：
   - 在扩散模型中取得巨大成功
   - 为VAE框架提供新的可能性
   - 推动了可控生成模型的发展

VAE-CG的成功证明了**引导生成**这一概念的有效性，为生成模型的发展提供了重要的思路和方法。

---

## 参考文献

[^1]: Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34, 8780-8794.

[^2]: Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*.

[^3]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 10684-10695.

[^4]: Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.

[^5]: Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models. *Advances in neural information processing systems*, 28.


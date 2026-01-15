# VAE-CG代码中Classifier Guidance原理详解

## 1. CG基本原理在代码中的体现

### 1.1 数学原理回顾

Classifier Guidance的核心数学公式：

$$z_{t+1} = z_t + \lambda \cdot \nabla_z \log p(y|z_t)$$

其中：
- $z_t$：第 $t$ 步的潜在变量
- $\lambda$：引导强度（guidance_scale）
- $\nabla_z \log p(y|z_t)$：分类器对目标类别 $y$ 的log概率关于 $z$ 的梯度

### 1.2 代码实现对应关系

**在 `sample_with_classifier_guidance` 方法中（216-297行）**：

#### 步骤1：初始化潜在变量（对应：从先验采样 $z_0$）
```python
# 第248行：从先验分布采样初始z
z = torch.randn(num_samples, self.latent_dim, device=device, requires_grad=True)
```
**数学对应**：$z_0 \sim \mathcal{N}(0, I)$

#### 步骤2：计算分类器的log概率（对应：$\log p(y|z_t)$）
```python
# 第257-258行：前向传播，计算分类器输出
logits = self.classifier(z)
log_probs = F.log_softmax(logits, dim=1)

# 第260-264行：计算目标类别的log概率
indices = torch.arange(num_samples, device=device).unsqueeze(1)
labels_expanded = c_or_labels.unsqueeze(1)
target_log_probs = log_probs.gather(1, labels_expanded).squeeze(1)
target_log_prob = target_log_probs.sum()
```
**数学对应**：
- `logits = classifier(z)` → 分类器输出
- `log_probs = log_softmax(logits)` → $\log p(y|z)$
- `target_log_prob` → 目标类别 $y$ 的 $\log p(y|z_t)$

#### 步骤3：计算梯度（对应：$\nabla_z \log p(y|z_t)$）
```python
# 第275-281行：计算梯度
grad = torch.autograd.grad(
    outputs=target_log_prob,
    inputs=z,
    create_graph=False,
    retain_graph=False,
    only_inputs=True
)[0]
```
**数学对应**：$\nabla_z \log p(y|z_t)$

这里使用了PyTorch的自动微分，计算 `target_log_prob` 对 `z` 的梯度。

#### 步骤4：更新潜在变量（对应：$z_{t+1} = z_t + \lambda \cdot \nabla_z \log p(y|z_t)$）
```python
# 第286行：更新z（核心公式）
z_new = z + guidance_scale * 0.01 * grad  # 小步长避免过度修正
```
**数学对应**：$z_{t+1} = z_t + \lambda \cdot \text{step_size} \cdot \nabla_z \log p(y|z_t)$

其中：
- `guidance_scale` = $\lambda$（引导强度）
- `0.01` = `step_size`（步长，避免过度修正）
- `grad` = $\nabla_z \log p(y|z_t)$

#### 步骤5：添加噪声（Langevin动力学）
```python
# 第288-289行：添加少量噪声保持探索
if step < 9:  # 最后一步不加噪声
    z_new = z_new + 0.01 * torch.randn_like(z_new)
```
**数学对应**：添加随机噪声项，保持探索性（类似Langevin动力学）

### 1.3 完整迭代过程（第250-292行）

```python
# 使用分类器梯度引导（Langevin动力学风格）
for step in range(10):  # 多次迭代以增强引导效果
    # 确保z需要梯度
    if not z.requires_grad:
        z = z.requires_grad_(True)
    
    # 前向传播
    logits = self.classifier(z)
    log_probs = F.log_softmax(logits, dim=1)
    
    # 计算目标类别的log概率
    target_log_probs = log_probs.gather(1, labels_expanded).squeeze(1)
    target_log_prob = target_log_probs.sum()
    
    # 计算梯度
    grad = torch.autograd.grad(...)[0]
    
    # 更新z
    with torch.no_grad():
        z_new = z + guidance_scale * 0.01 * grad
        if step < 9:
            z_new = z_new + 0.01 * torch.randn_like(z_new)
        z = z_new.detach().requires_grad_(True)
```

**数学对应**：
$$z_{t+1} = z_t + \lambda \cdot \eta \cdot \nabla_z \log p(y|z_t) + \epsilon$$

其中 $\eta = 0.01$ 是步长，$\epsilon$ 是噪声项。

---

## 2. 训练过程的两阶段

### 2.1 理论上的两阶段训练

理论上，CG需要**两阶段训练**：

**阶段1：训练VAE模型**
- 训练编码器：$q_\phi(z|x, c)$
- 训练解码器：$p_\theta(x|z, c)$
- 损失函数：$\mathcal{L}_{VAE} = \text{ReconLoss} + \text{KL}$

**阶段2：训练分类器**
- 训练潜在空间分类器：$p_\psi(y|z)$
- 使用VAE编码器得到的潜在变量 $z$ 和标签 $y$
- 损失函数：$\mathcal{L}_{classifier} = \text{CrossEntropy}$

### 2.2 代码中的实际实现

**当前代码使用的是联合训练（Joint Training）**，而不是严格的两阶段训练：

#### 2.2.1 模型结构（第165-190行）

```python
class VAE_CG(nn.Module):
    def __init__(self, ...):
        # VAE组件：使用条件
        self.encoder = Encoder(...)      # 阶段1：VAE的编码器
        self.decoder = Decoder(...)      # 阶段1：VAE的解码器
        
        # 潜在空间分类器（用于CG）
        self.classifier = LatentClassifier(...)  # 阶段2：分类器
```

**对应**：
- `encoder` + `decoder` = 阶段1的VAE模型
- `classifier` = 阶段2的分类器

#### 2.2.2 前向传播（第192-214行）

```python
def forward(self, x, c_or_labels=None):
    # 编码条件
    c = self.condition_encoder(c_or_labels)
    
    # 编码（阶段1：VAE编码）
    mu, sigma = self.encoder(x, c)
    z = self._reparameterize(mu, sigma)
    
    # 解码（阶段1：VAE解码）
    x_hat = self.decoder(z, c)
    
    # 分类器预测（阶段2：分类器）
    class_logits = self.classifier(z)
    
    return x_hat, mu, sigma, class_logits
```

**对应**：
- `encoder` → `decoder` → `x_hat` = 阶段1的VAE前向传播
- `classifier(z)` → `class_logits` = 阶段2的分类器前向传播

#### 2.2.3 损失函数（第416-436行）

```python
def loss_function_cg(x_hat, x, mu, sigma, class_logits, labels, classification_weight=0.1):
    batch_size = x.shape[0]
    
    # 重构损失（阶段1：VAE损失的一部分）
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度（阶段1：VAE损失的一部分）
    kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(sigma + 1e-8) - mu ** 2 - sigma ** 2)
    
    # 分类损失（阶段2：分类器损失）
    classification_loss = F.cross_entropy(class_logits, labels, reduction='sum')
    
    # 总损失 = VAE损失 + 分类损失
    total_loss = recon_loss + kl_loss + classification_weight * classification_loss
    
    return total_loss / batch_size
```

**数学对应**：
$$\mathcal{L}_{total} = \mathcal{L}_{VAE} + \alpha \cdot \mathcal{L}_{classifier}$$

其中：
- $\mathcal{L}_{VAE} = \text{ReconLoss} + \text{KL}$ = 阶段1的损失
- $\mathcal{L}_{classifier} = \text{CrossEntropy}$ = 阶段2的损失
- $\alpha = \text{classification_weight}$ = 分类损失的权重（默认0.1）

#### 2.2.4 训练循环（train_infer.py 第100-120行）

```python
for x, label in pbar:
    x = x.view(-1, input_dim).to(device)
    labels = label.to(device)
    
    if model_type == 'vae_cg':
        # 前向传播：同时计算VAE和分类器输出
        x_hat, mu, sigma, class_logits = model(x, labels)
        
        # 计算总损失（包含VAE损失和分类损失）
        loss = loss_function_cg(x_hat, x, mu, sigma, class_logits, labels, classification_weight)
    
    # 反向传播：同时更新VAE和分类器的参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**对应**：
- 一次前向传播同时得到VAE输出和分类器输出
- 一次反向传播同时更新所有参数（encoder、decoder、classifier）

### 2.3 两阶段 vs 联合训练

#### 严格的两阶段训练（理论）：

```python
# 阶段1：只训练VAE
for epoch in range(epochs_stage1):
    for x, labels in train_loader:
        x_hat, mu, sigma = model.encoder_decoder_forward(x, labels)
        loss = vae_loss(x_hat, x, mu, sigma)
        loss.backward()
        optimizer_VAE.step()

# 阶段2：冻结VAE，只训练分类器
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = False

for epoch in range(epochs_stage2):
    for x, labels in train_loader:
        _, _, _, z = model.encode(x, labels)
        class_logits = model.classifier(z)
        loss = classification_loss(class_logits, labels)
        loss.backward()
        optimizer_classifier.step()
```

#### 当前代码的联合训练（实际）：

```python
# 联合训练：同时训练VAE和分类器
for epoch in range(epochs):
    for x, labels in train_loader:
        x_hat, mu, sigma, class_logits = model(x, labels)
        loss = vae_loss + alpha * classification_loss
        loss.backward()
        optimizer.step()  # 同时更新所有参数
```

### 2.4 为什么联合训练也可以工作？

1. **目标一致**：VAE学习潜在表示，分类器在潜在表示上分类，两者互补
2. **参数共享**：分类器使用VAE编码器得到的 $z$，共享潜在空间
3. **权重平衡**：通过 `classification_weight` 控制分类损失的权重，避免分类损失过强影响VAE训练

### 2.5 如果要实现严格的两阶段训练

可以这样修改代码：

```python
# 阶段1：训练VAE
print("阶段1：训练VAE...")
for epoch in range(epochs_stage1):
    for x, labels in train_loader:
        # 只计算VAE损失
        x_hat, mu, sigma, _ = model(x, labels)
        loss = loss_function_cfg(x_hat, x, mu, sigma)  # 只用VAE损失
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 阶段2：冻结VAE参数，训练分类器
print("阶段2：训练分类器...")
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = False

for epoch in range(epochs_stage2):
    for x, labels in train_loader:
        # 只计算分类损失
        _, _, _, class_logits = model(x, labels)
        loss = F.cross_entropy(class_logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 只更新分类器参数
```

---

## 3. 总结

### 3.1 CG基本原理的代码对应

| 数学公式 | 代码位置 | 说明 |
|---------|---------|------|
| $z_0 \sim \mathcal{N}(0, I)$ | 第248行 | 从先验采样初始z |
| $\log p(y\|z_t)$ | 第257-264行 | 计算分类器的log概率 |
| $\nabla_z \log p(y\|z_t)$ | 第275-281行 | 计算梯度 |
| $z_{t+1} = z_t + \lambda \cdot \nabla_z \log p(y\|z_t)$ | 第286行 | 更新z |

### 3.2 训练过程

| 阶段 | 组件 | 损失函数 | 代码位置 |
|-----|------|---------|---------|
| **阶段1（概念）** | encoder + decoder | 重构损失 + KL散度 | `loss_function_cg` 中的 `recon_loss + kl_loss` |
| **阶段2（概念）** | classifier | 分类损失 | `loss_function_cg` 中的 `classification_loss` |
| **实际实现** | 所有组件联合训练 | 总损失 = VAE损失 + α × 分类损失 | 第416-436行 |

### 3.3 关键点

1. **CG的核心**：在 `sample_with_classifier_guidance` 中实现梯度引导算法
2. **训练方式**：当前是联合训练，通过权重平衡两种损失
3. **灵活性**：可以改为严格的两阶段训练，但联合训练通常效果也很好



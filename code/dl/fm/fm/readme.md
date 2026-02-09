# Flow Matching 概念梳理

## 1. 你写的是不是「最简单的 FM」？

**是的。** 当前实现是**最基础**的 Flow Matching：

- **概率路径**：线性插值 $x_t = (1-t)x_0 + t x_1$（$x_0\sim\mathcal{N}(0,I)$，$x_1$ 为数据）
- **目标**：条件向量场 $u_t(x|x_1) = x_1 - x_0$（常数），训练时用 **Conditional Flow Matching (CFM)** 目标
- **采样**：从 $x(0)\sim\mathcal{N}(0,I)$ 解 ODE $\mathrm{d}x/\mathrm{d}t = v_\theta(x,t)$，欧拉法

下面把「边缘场、条件场、条件 FM、Normalizing Flows、不同路径、Guidance」等概念理清。

---

## 2. 核心概念对应关系

| 中文（常见说法） | 英文 | 含义 |
|------------------|------|------|
| 概率路径 | probability path | 随时间 $t\in[0,1]$ 从噪声 $p_0$ 变到数据 $p_1$ 的分布族 $p_t(x)$；FM 要学的流就是让样本沿这条路径演化。 |
| 边缘向量场 / 边缘场 | marginal vector field | 驱动整条概率路径 $p_t$ 的「总」速度场 $u_t(x)$（满足连续性方程）。理论上是我们想逼近的目标，但需对 $x_1$ 积分才有，无闭式，故不直接优化。 |
| 条件向量场 / 条件场 | conditional vector field | 固定终点 $x_1$ 时，从噪声到 $x_1$ 这一条路径上的速度场 $u_t(x\!\mid\!x_1)$。每条路径有闭式（如线性时为 $x_1-x_0$），训练时用它监督模型。 |
| 条件流匹配 | Conditional Flow Matching (CFM) | 不直接优化边缘场，而是最小化「模型速度 vs 条件场」的 MSE；在期望下与该边缘目标梯度一致，且训练时**不需仿真 ODE**（simulation-free）。 |
| 归一化流 / 连续归一化流 | Normalizing Flows (NF) / Continuous NF (CNF) | NF：可逆映射把简单分布变到数据分布；CNF：用 ODE 的连续版本。**FM 训练的就是 CNF 的向量场** $v_\theta$，但用 CFM 目标而非似然，训练阶段不解 ODE。 |

---

## 2.1 两种「条件」辨析（易混点）

文档里会出现两个不同的「条件」，**不是同一概念**：

| 说法 | 条件指谁 | 含义 | 典型记号 |
|------|----------|------|----------|
| **条件向量场** / CFM 里的「条件」 | **终点样本** $x_1$ | 训练时的数学构造：固定「从噪声到哪个数据点 $x_1$」这条路径，该路径上的速度场叫 $u_t(x\!\mid\!x_1)$。用来写可算的 CFM 目标，**不是用户指定的生成条件**。 | $u_t(x\!\mid\!x_1)$ |
| **条件生成** / 条件 FM 里的「条件」 | **用户给定的条件** $y$ | 任务层面：在「给定类别、文本、观测等 $y$」下生成 $p(x\!\mid\!y)$。模型输入里会多一个 $y$，即 $v_\theta(x,t,y)$。 | $p(x\!\mid\!y)$，$v_\theta(x,t,y)$ |

- **条件场**：每个**数据点** $x_1$ 对应一条路径，$x_1$ 是「路径的终点」，训练时从数据里采样得到，推理时**不**给定。
- **条件生成**：**用户**在推理时给定 $y$（如「生成数字 7」），模型据此生成符合 $y$ 的样本。

---

## 3. 边缘场 vs 条件场（数学）

### 3.1 边缘向量场 $u_t(x)$

- 定义：希望学一个 $v_\theta(x,t)$，使得由 ODE $\frac{\mathrm{d}x}{\mathrm{d}t} = v_\theta(x,t)$ 生成的分布 $p_t$ 满足 $p_0=$ 噪声、$p_1=$ 数据。
- 等价地，存在一个**目标**向量场 $u_t(x)$，满足连续性方程：

$$
\frac{\partial p_t(x)}{\partial t} = -\nabla_x\cdot\big(u_t(x)\,p_t(x)\big).
$$

- **Flow Matching 的「理想目标」**就是让 $v_\theta \approx u_t$。但 $u_t(x)$ 是**边缘化**出来的，一般**没有闭式、难算**，所以不直接优化它。

### 3.2 条件向量场 $u_t(x|x_1)$

- 对**每个**目标样本 $x_1$，定义一条**条件概率路径** $p_t(x|x_1)$（例如：$t=0$ 时是噪声，$t=1$ 时集中在 $x_1$）。
- 这条路径对应一个**条件向量场** $u_t(x|x_1)$，满足：

$$
\frac{\partial p_t(x|x_1)}{\partial t} = -\nabla_x\cdot\big(u_t(x|x_1)\,p_t(x|x_1)\big).
$$

- **关键性质**：边缘场可以由条件场「边际化」得到：

$$
u_t(x) = \int u_t(x|x_1)\,\frac{p_t(x|x_1)\,q(x_1)}{p_t(x)}\,\mathrm{d}x_1.
$$

也就是说：**边缘场 = 对所有 $x_1$ 的条件场按 $p_t(x|x_1)q(x_1)/p_t(x)$ 加权平均**。

### 3.3 条件流匹配 (CFM)：用条件目标代替边缘目标

- **边缘 FM 目标**（难算）：

$$
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t,\,x\sim p_t}\big[\|v_\theta(x,t) - u_t(x)\|^2\big].
$$

- **条件 FM 目标**（可算）：

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\,x_1\sim q(x_1),\,x\sim p_t(x|x_1)}\big[\|v_\theta(x,t) - u_t(x|x_1)\|^2\big].
$$

- **定理**（Lipman et al.）：在期望意义下，**CFM 目标给出的梯度与边缘 FM 目标一致**。因此：
  - 我们**实际训练**用的是「条件目标」$\mathcal{L}_{\mathrm{CFM}}$；
  - 学到的 $v_\theta$ 在分布意义下逼近的是**边缘场** $u_t(x)$；
  - 不需要仿真 ODE、不需要算 $p_t(x)$，所以是 **simulation-free**。

你写的代码就是在做 CFM：对每条「线」$(x_0, x_1)$ 用 $u_t(x|x_1)=x_1-x_0$ 做监督，这就是**条件场**；等价地是在逼近**边缘场**。

---

## 4. 概率路径的几种选择（Normalizing Flows、OT、扩散等）

「路径」指的是：$p_t(x|x_1)$（以及 $\mu_t(x_1),\sigma_t(x_1)$ 等）如何随 $t$ 变化。不同选择得到不同的 $u_t(x|x_1)$ 和不同的训练/采样性质。

### 4.1 线性插值（你当前用的）

- $x_t = (1-t)x_0 + t x_1$，即 $p_t(x|x_1)$ 是 $x_0\sim p_0$ 与 $x_1$ 之间的线性插值。
- 条件速度：$u_t(x|x_1) = x_1 - x_0$（常数，与 $t$ 无关）。
- **优点**：实现最简单，训练稳定。**缺点**：$(x_0,x_1)$ 随机配对时路径会交叉，学到的**边缘**场是「平均方向」，轨迹较弯，采样时常需较多步。

### 4.2 最优传输 (OT) 路径 / 位移插值

- 先对噪声与数据做**配对**（如 minibatch 内 OT 或最优传输），再在**配对**的 $(x_0,x_1)$ 之间做线性插值（或 OT 位移插值）。
- 路径更直、更少交叉，采样步数可更少，甚至 1–4 步；但需要算 OT 或近似（如 minibatch OT、sinkhorn），实现更重。
- 有些工作叫 **Optimal Transport Flow Matching** 或 **Rectified Flow**（见下）。

### 4.3 扩散型路径 (Gaussian conditional path)

- $p_t(x|x_1) = \mathcal{N}(x;\,\mu_t(x_1),\,\sigma_t^2(x_1)I)$，其中 $\mu_t(x_1)、\sigma_t(x_1)$ 按扩散 schedule 设计。
- 条件向量场有闭式（见 HackMD / Lipman 论文）：

$$
u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu_t'(x_1).
$$
- 更接近 DDPM 的边际分布，适合从扩散模型迁移；路径一般比线性插值更弯。

### 4.4 Rectified Flow（「拉直」的流）

- **问题**：用随机配对的线性插值训练时，学到的**边缘**场是平均场，轨迹弯曲，需要很多步才能采好样。
- **做法**：  
  1）用当前模型从噪声生成样本，得到「模型意义上的配对」$(x_0, \hat{x}_1)$；  
  2）在这些**直连**的配对之间再训练线性 FM；  
  3）重复多轮，路径越来越直，最终可 1-step 或 few-step 采样。
- 可以理解为：在**迭代地**逼近一条「更直」的、类似 OT 的路径。

### 4.5 Normalizing Flows（归一化流）与 Flow Matching 的关系

- **Normalizing Flows（NF）**：用可逆变换 $f_\theta$ 把简单分布 $p_0$（如高斯）映射到数据分布 $p_1$，通过变量替换可算 exact log-likelihood。传统 NF 是**离散**的若干层 $x_{k+1} = f_k(x_k)$。
- **Continuous Normalizing Flows（CNF）**：把离散层变成**连续时间**的 ODE，$x_t$ 由 $\frac{\mathrm{d}x}{\mathrm{d}t} = v_\theta(x,t)$ 从 $t=0$ 流到 $t=1$。理论上也可算 log-likelihood（用 instantaneous change of variables），但训练时需要**沿轨迹做 ODE 仿真**，成本高。
- **Flow Matching 和 CNF**：FM 是**训练 CNF 的一种方式**——不直接优化 likelihood，而是用 **simulation-free** 的目标（CFM）去拟合向量场 $v_\theta \approx u_t$，训练时**不需要**解 ODE；采样时再用学到的 $v_\theta$ 解 ODE，等价于用 CNF 生成样本。所以：**FM 学的是（连续）归一化流的向量场**，只是训练目标从「似然」换成了「匹配条件速度场」。

---

## 5. 条件 FM 与 Guidance（条件生成、引导）

这里的「条件」指**用户给定的条件** $y$（类别、文本、观测等），即**条件生成**；与前面 **条件向量场** 里的「条件」$x_1$（路径终点、训练构造）不是一回事，见 **2.1 两种「条件」辨析**。

### 5.1 条件生成：$p(x|y)$

- 目标：从 $p_0$ 出发，采样结束时得到的是 $p(x_1|y)$，而不是无条件 $p(x_1)$。
- 做法通常有两类：  
  - **训练时**就带条件：$v_\theta(x,t,y)$，数据是 $(x,y)$；  
  - 或**训练时**用 classifier-free：随机 drop 条件，学 $v_\theta(x,t,y)$ 与 $v_\theta(x,t,\varnothing)$，推理时用 guidance 组合。

### 5.2 Classifier-Free Guidance（无分类器引导）

- 和扩散里一样：训练时以一定概率把条件 $y$ 换成「空」$\varnothing$，得到 $v_\theta(x,t,y)$ 和 $v_\theta(x,t,\varnothing)$。
- 推理时用**引导速度**：

$$
\tilde{v}(x,t) = v_\theta(x,t,\varnothing) + w\cdot\big(v_\theta(x,t,y) - v_\theta(x,t,\varnothing)\big),
$$

$w>1$ 时更贴合条件 $y$（更强引导）。
- 这类工作有时叫 **Guided Flows**，把扩散里的 CFG 搬到 FM 的**速度场**上。

### 5.3 其他 Guidance（后验、Doob h-transform）

- **后验采样**：若有一个观测模型 $p(y|x_1)$，希望从 $p(x_1|y)$ 采样，可以构造**条件向量场** $\tilde{u}_t(x|y)$，形式上是 $u_t(x)$ 加上一个与 $\nabla_x\log p_t(y|x)$ 有关的项（类似 Doob h-transform），用学到的 $v_\theta$ 和近似 score 来近似。
- 这些属于「在已有 FM 模型上做条件/引导」，而不是基础 FM 本身。

---

## 6. 概念小结（对应你的问题）

| 你问的 | 对应概念 | 简要说明 |
|--------|----------|----------|
| Normalizing Flows（归一化流）| NF / CNF | 用可逆/ODE 变换把简单分布变到数据分布；FM 训练的就是 CNF 的向量场，只是用 CFM 目标、simulation-free |
| 条件向量场 | $u_t(x\!\mid\!x_1)$ | 给定终点 $x_1$ 的那条路径上的速度；有闭式，用来写 CFM 目标 |
| 边缘场 | $u_t(x)$ | 由 $p_t$ 连续性方程决定的「总」速度场；难算，用 CFM 间接学 |
| 条件 FM | CFM | 用条件目标 $\mathbb{E}[\|v_\theta - u_t(x\!\mid\!x_1)\|^2]$ 训练，梯度等价于边缘 FM |
| 条件 FM（条件生成）| $p(x\!\mid\!y)$ + CFG | 带条件 $y$ 的生成；常配合 classifier-free guidance |
| Guidance | 引导 | 在速度场上加一项（如 CFG、或 $\nabla\log p(y\!\mid\!x)$）以加强条件或后验 |

---

## 7. 你当前的实现在这张图里的位置

- **路径**：线性插值（最基础）。
- **训练目标**：CFM（条件场 $u_t(x|x_1)=x_1-x_0$）。
- **没有**：OT 配对、扩散路径、Rectified Flow、条件生成、Guidance。

若要扩展，可以按这个顺序加：

1. **换路径**：例如 Gaussian (扩散) 路径、或 minibatch OT + 线性插值。  
2. **Rectified Flow**：用当前模型生成配对，再在这些配对上重新训练线性 FM。  
3. **条件生成**：输入加条件 $y$，训练 $v_\theta(x,t,y)$，推理时用 CFG。

如果你愿意，我可以按「最小改动」的方式，在现有代码里加一个「条件 FM + CFG」的版本（例如 MNIST 类别条件），并标清哪里是条件场、哪里是 guidance。
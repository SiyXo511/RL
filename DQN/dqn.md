
# DQN 笔记

## 核心概念

- **目标**：使用深度神经网络近似 Q 函数 $Q(s,a;\theta)$，替代 Q 表以处理高维或连续状态空间。
- **关键思想**：
  - 用神经网络预测每个动作的 Q 值
  - 使用经验回放（Replay Buffer）打破样本相关性
  - 使用目标网络（Target Network）稳定训练过程
- **策略类型**：**Off-policy（离策略）** - 学习最优策略，与 Q-Learning 相同

## 为什么需要 DQN？

传统 Q-Learning 依赖 Q 表，只适合状态空间较小、离散的任务。像 CartPole、Atari 游戏、机器人控制等任务的状态维度高、连续，Q 表无法直接应用。DQN 用神经网络近似 Q 函数，可对任意状态输入输出动作价值。

### Q-Learning vs DQN 的局限性对比

| 特性 | Q-Learning | DQN |
|------|-----------|-----|
| **存储方式** | Q 表（表格） | 神经网络（函数近似） |
| **状态空间** | 离散、小规模（如 16 个状态） | 连续、大规模（如 4 维连续状态） |
| **泛化能力** | 无（每个状态独立学习） | 有（相似状态共享知识） |
| **内存需求** | $O(|S| \times |A|)$ | $O(\text{网络参数})$ |
| **适用场景** | FrozenLake（16 状态） | CartPole（连续状态）、Atari（图像） |

## 数学基础：DQN 的损失函数

DQN 的核心是使用神经网络近似 Q 函数，通过最小化时序差分误差来学习。

### 目标函数

DQN 的目标是学习一个神经网络 $Q(s,a;\theta)$，使得它尽可能接近真实的 Q 函数。

**损失函数**：
$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( y - Q(s,a;\theta) \right)^2 \right]
$$

其中目标值 $y$ 为：
$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

- $\theta$：策略网络的参数（不断更新）
- $\theta^-$：目标网络的参数（定期更新）
- $D$：经验回放缓冲区中的经验分布

### 为什么需要目标网络？

如果使用同一个网络计算当前 Q 值和目标 Q 值：
$$
y = r + \gamma \max_{a'} Q(s', a'; \theta)
$$

目标值 $y$ 会随着 $\theta$ 的更新而不断变化，导致：
1. **目标不稳定**：每次更新参数后，目标值也在变化
2. **训练发散**：网络试图追逐一个移动的目标，难以收敛

**解决方案**：使用目标网络 $\theta^-$，定期（如每 500 步）从策略网络复制参数：
$$
\theta^- \leftarrow \theta
$$

这样目标值在一段时间内保持稳定，训练更稳定。

## DQN 算法框架

1. **策略网络（Policy Net）**：学习当前 Q 函数，用于选择动作、计算当前 Q 值。
2. **目标网络（Target Net）**：定期复制策略网络参数，用于计算 TD 目标，避免目标值快速变化。
3. **Replay Buffer**：存储 $(s, a, r, s', done)$，随机采样形成训练批次，减少样本间的相关性，提高样本利用率。

## 算法流程

1. 初始化策略网络和目标网络，目标网络参数拷贝自策略网络。
2. 初始化经验回放缓冲区。
3. 对每个 episode：
   - 从环境 reset 获取初始状态 $s$。
   - 在每个时间步：
     1. 使用 ε-greedy 策略选择动作 $a$。
     2. 执行动作，得到 $(s', r, done)$，存入 Replay Buffer。
     3. 如果 Replay Buffer 样本数超过阈值，随机采样一个 minibatch 进行训练：
        - 当前 Q 值：$Q(s,a;\theta)$（策略网络输出）
        - 目标 Q 值：$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ （目标网络输出，若 done 则只有 $r$）
        - 用 MSELoss 计算 $Q(s,a;\theta)$ 与 $y$ 的差距，反向传播更新策略网络参数。
     4. 每隔固定步数，将策略网络参数拷贝到目标网络。
   - 记录奖励、ε 值等指标。

## 关键模块解释

### 1. ε-greedy 动作选择

```python
epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
if random.random() < epsilon:
    action = env.action_space.sample()
else:
    action = policy_net(state).argmax()
```

- ε 随训练步数指数衰减。
- 训练初期高探索，后期逐步利用。

### 2. Replay Buffer

```python
memory = deque(maxlen=buffer_capacity)
memory.append((state, action, reward, next_state, done))
batch = random.sample(memory, batch_size)
```

- 储存大量经验，随机采样形成无序批次。
- 可多次利用同一经验，提升样本效率。

### 3. 策略网络 & 目标网络

```python
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
```

- 策略网络用于选择动作和计算当前 Q 值。
- 目标网络用于计算目标 Q 值，参数更新频率低（例如每 500 步一次）。

### 4. 训练环节（optimize_model）

```python
states, actions, rewards, next_states, dones = memory.sample(batch_size)
q_values = policy_net(states).gather(1, actions)
with torch.no_grad():
    max_next_q = target_net(next_states).max(1, keepdim=True)[0]
    target_q = rewards + gamma * (1 - dones) * max_next_q
loss = mse_loss(q_values, target_q)
```

- `gather(1, actions)`：取出每个样本对应动作的 Q 值。
- `target_q`：即时奖励 + 折扣未来奖励（若 done 则未来奖励为 0）。
- 使用 MSELoss 约束当前 Q 值向目标 Q 值靠近。
- 训练结束后，定期同步目标网络参数。

### 5. 训练循环

```python
for episode in range(max_episodes):
    state, info = env.reset()
    for t in range(max_steps):
        action, epsilon = select_action(state, global_step)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        memory.push(state, action, reward, next_state, done)
        state = next_state
        ...
        if global_step > start_training_after:
            optimize_model()
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
```

- 每步更新全局步数 `global_step`，用于 ε 衰减和参数同步。
- `start_training_after` 确保先收集足够经验再训练。
- 训练过程中记录奖励和 ε 变化，用于可视化。

## 可视化与评估

- **训练曲线**：绘制每轮奖励、移动平均奖励、ε 衰减曲线。
- **评估**：使用训练好的策略，在 `render_mode="human"` 环境中演示，不再探索。
- **窗口调整**：`resize_cartpole_window()` 将 CartPole 渲染窗口放大，便于观察。

## DQN 的关键理解

### 1. 更新 Q 值时：使用"假设最优"（与 Q-Learning 相同）

**时机**：从经验回放缓冲区采样一批经验后，计算损失并更新网络。

**过程**：
- DQN 使用目标网络计算下一状态的最大 Q 值：$\max_{a'} Q(s', a'; \theta^-)$
- 这假设在状态 $s'$ 总是选择最优动作
- 使用这个**最大 Q 值**来计算目标值并更新策略网络

**关键点**：
- 与 Q-Learning 相同，DQN 也是 **Off-policy** 算法
- 学习的是最优策略，而不是执行策略
- 目标网络提供稳定的目标值，避免训练不稳定

### 2. 实际行动时：使用 ε-greedy 策略（真正执行）

**时机**：智能体在环境中需要选择动作时。

**策略**：使用 $\varepsilon$-greedy 策略：
- **利用（Exploit）**：以 $1-\varepsilon$ 的概率，选择 $Q(s,a;\theta)$ 最大的动作
- **探索（Explore）**：以 $\varepsilon$ 的概率，随机选择动作

**关键点**：
- 实际执行的动作**不一定**是 Q 值最大的动作
- 探索时可能选择任意动作，这是为了发现更好的策略
- 评估阶段（`render_mode="human"`）时，通常设置 $\varepsilon = 0$，纯贪心选择

### 总结

| 阶段 | 使用的值 | 是否实际执行 | 目的 |
|------|---------|------------|------|
| **更新 Q 值** | $\max_{a'} Q(s', a'; \theta^-)$（目标网络） | ❌ 不执行 | 学习最优策略 |
| **选择动作** | $\varepsilon$-greedy（策略网络） | ✅ 实际执行 | 平衡探索与利用 |

## 重要超参数

### 基础参数

- **学习率（learning_rate）**：
  - **推荐值**：$10^{-3}$ 到 $10^{-4}$（如 `1e-3`）
  - **作用**：控制神经网络参数更新的步长
  - **调参建议**：
    - 太大：训练不稳定，可能发散
    - 太小：收敛太慢，训练时间长

- **折扣因子（gamma）**：
  - **推荐值**：0.99（接近 1）
  - **作用**：衡量未来奖励的重要性
  - **说明**：对于 CartPole 这种需要长期平衡的任务，保持高折扣因子很重要

### 网络训练参数

- **批次大小（batch_size）**：
  - **推荐值**：32-128（如 64）
  - **作用**：每次训练使用的样本数量
  - **调参建议**：
    - 太小（< 32）：梯度估计噪音大，训练不稳定
    - 太大（> 256）：内存占用大，更新频率低

- **经验回放缓冲区容量（buffer_capacity）**：
  - **推荐值**：10,000-100,000（如 50,000）
  - **作用**：存储历史经验的最大数量
  - **说明**：越大样本多样性越好，但内存占用也越大

- **开始训练阈值（start_training_after）**：
  - **推荐值**：1,000-10,000
  - **作用**：在开始训练前需要收集的经验数量
  - **原因**：避免网络过早拟合噪声样本

### 目标网络参数

- **目标网络更新频率（target_update_freq）**：
  - **推荐值**：100-1000 步（如 500）
  - **作用**：每隔多少步将策略网络参数复制到目标网络
  - **调参建议**：
    - 太频繁（< 100）：目标值变化太快，训练不稳定
    - 太慢（> 2000）：目标网络老化，学习效率低

### 探索策略参数

- **初始探索率（epsilon_start）**：
  - **推荐值**：1.0
  - **作用**：训练开始时的探索率，保证充分探索

- **最终探索率（epsilon_end）**：
  - **推荐值**：0.01-0.1（如 0.05）
  - **作用**：训练后期的探索率，保留少量探索避免完全贪心

- **探索率衰减步数（epsilon_decay）**：
  - **推荐值**：1,000-10,000（如 5,000）
  - **作用**：控制探索率从初始值衰减到最终值的速度
  - **公式**：$\varepsilon = \varepsilon_{\text{end}} + (\varepsilon_{\text{start}} - \varepsilon_{\text{end}}) \times e^{-\text{step} / \text{decay}}$

### 训练规模参数

- **最大训练轮数（max_episodes）**：
  - **推荐值**：根据任务复杂度，CartPole 通常 500-2000 轮
  - **作用**：控制训练的总轮数

- **每轮最大步数（max_steps）**：
  - **推荐值**：根据任务，CartPole 通常 500 步
  - **作用**：每个 episode 的最大步数限制

## 代码要点（`dqn_cartpole.py`）

### 核心实现

- **经验回放缓冲区**：使用 `deque` 实现，对应代码第 40-63 行
  ```python
  class ReplayBuffer:
      def __init__(self, capacity):
          self.buffer = deque(maxlen=capacity)
  ```

- **DQN 网络结构**：三层全连接网络，对应代码第 68-81 行
  ```python
  class DQN(nn.Module):
      def __init__(self, input_dim, output_dim):
          self.net = nn.Sequential(
              nn.Linear(input_dim, 128), nn.ReLU(),
              nn.Linear(128, 128), nn.ReLU(),
              nn.Linear(128, output_dim)
          )
  ```

- **训练函数**：对应代码第 114-138 行
  ```python
  def optimize_model():
      # 采样经验
      states, actions, rewards, next_states, dones = memory.sample(batch_size)
      # 当前Q值
      q_values = policy_net(states).gather(1, actions)
      # 目标Q值（使用目标网络）
      with torch.no_grad():
          max_next_q = target_net(next_states).max(1, keepdim=True)[0]
          target_q = rewards + gamma * (1 - dones) * max_next_q
      # 计算损失并更新
      loss = mse_loss(q_values, target_q)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
      optimizer.step()
  ```

- **动作选择**：对应代码第 95-111 行
  ```python
  def select_action(state, step):
      epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
      if random.random() < epsilon:
          return env.action_space.sample(), epsilon  # 探索
      else:
          return policy_net(state).argmax().item(), epsilon  # 利用
  ```

### CartPole 环境配置

```python
# 环境设置
env_id = "CartPole-v1"
state_dim = 4   # 小车位置、速度、杆角度、角速度
action_dim = 2  # 向左推、向右推

# 超参数（针对 CartPole 优化）
learning_rate = 1e-3
gamma = 0.99
batch_size = 64
buffer_capacity = 50_000
target_update_freq = 500
```

### 实用功能

- **可视化训练过程**：绘制奖励曲线和 ε 衰减曲线
- **窗口调整**：`resize_cartpole_window()` 函数调整 CartPole 渲染窗口大小
- **评估模式**：评估时使用纯贪心策略（$\varepsilon = 0$），不再探索

## 常见问题与调参建议

### 训练效果差（奖励不上升）

**症状**：训练过程中奖励曲线几乎是一条水平线，智能体无法学习。

**原因分析**：
1. **经验池不足**：`start_training_after` 太小，网络过早拟合噪声
2. **学习率过低**：参数更新太慢，无法有效学习
3. **探索不足**：$\varepsilon$ 衰减太快，智能体过早停止探索
4. **目标网络更新过慢**：目标值老化，学习效率低

**解决方案**：

| 参数 | 问题值 | 推荐值 | 原因 |
|------|--------|--------|------|
| 学习率 | < 1e-4 | **1e-3** | 加快参数更新速度 |
| start_training_after | < 500 | **1,000+** | 确保有足够经验再训练 |
| epsilon_decay | < 1,000 | **5,000+** | 给智能体充足探索时间 |
| target_update_freq | > 2,000 | **500** | 保持目标值相对新鲜 |

### 训练不稳定（奖励波动大）

**症状**：奖励曲线波动剧烈，训练过程不稳定。

**解决方案**：
1. **梯度裁剪**：代码中已启用 `clip_grad_norm_(..., 1.0)`
2. **降低学习率**：从 `1e-3` 降到 `5e-4` 或 `1e-4`
3. **增大批次大小**：从 64 增加到 128 或 256
4. **调整目标网络更新频率**：从 500 调整到 1000

### 探索不足

**症状**：训练后期奖励不再提升，可能陷入局部最优。

**解决方案**：
1. **放慢 ε 衰减**：增大 `epsilon_decay`（如从 5,000 到 10,000）
2. **提高最小 ε**：从 0.05 提高到 0.1
3. **检查探索率曲线**：确保训练后期仍有足够的探索

### 评估表现不佳

**症状**：训练时奖励很高，但评估时表现很差。

**可能原因**：
1. **过拟合**：网络过度拟合训练数据
2. **探索不足**：训练后期探索太少，策略不够鲁棒

**解决方案**：
1. **增加训练轮数**：给网络更多时间学习
2. **提高最小 ε**：在训练后期保持一定探索
3. **检查网络容量**：如果网络太大，考虑减小隐藏层大小

## 与 Q-Learning/SARSA 的对比

| 特性 | Q-Learning/SARSA | DQN |
|------|------------------|-----|
| 状态表示 | 表格 | 神经网络 |
| 经验利用 | 单步在线更新 | 批量更新 + 经验回放 |
| 目标值 | 直接来自 Q 表 | 来自目标网络 |
| 适用场景 | 小状态空间 | 大状态空间、连续状态 |
| 探索策略 | ε-greedy | 同样 ε-greedy，但值来自神经网络 |

## 记忆要点

1. **双网络结构**：策略网络负责行动和计算当前 Q 值，目标网络提供稳定的目标值，避免训练不稳定。

2. **经验回放**：随机采样历史经验，打破时间相关性，提高样本利用率，这是 DQN 成功的关键之一。

3. **Off-policy 学习**：与 Q-Learning 相同，DQN 学习最优策略，使用"假设最优"的目标值更新。

4. **ε-greedy 探索**：训练全程使用 ε-greedy，评估时纯贪心（$\varepsilon = 0$）。

5. **梯度裁剪**：防止训练不稳定（代码中已启用 `clip_grad_norm_`）。

6. **超参数调优**：
   - 学习率：通常 $10^{-3}$ 到 $10^{-4}$
   - 批次大小：32-128
   - 目标网络更新频率：100-1000 步
   - 探索率衰减：根据任务调整

7. **可视化与评估**：通过奖励曲线、ε 衰减曲线、屏幕演示查看学习效果。

8. **适用场景**：DQN 特别适合状态空间大、连续的任务，如 CartPole、Atari 游戏等。

以上笔记对应 `dqn_cartpole.py` 的实现，可作为理解 DQN 流程和代码结构的参考。


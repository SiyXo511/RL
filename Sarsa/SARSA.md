# SARSA 笔记

## 核心概念

- **目标**：学习当前执行策略的值函数 $Q(s,a)$，估计在状态 $s$ 采取动作 $a$ 的长期价值。
- **策略类型**：**On-policy（同策略）** - 学习的是当前执行策略，而不是最优策略。
- **价值传播**：通过贝尔曼方程，使用**实际执行的下一个动作**来更新Q值。
- **策略**：使用 ε-greedy，平衡探索与利用。

## SARSA vs Q-Learning

### 核心区别

| 特性 | SARSA | Q-Learning |
|------|-------|------------|
| **策略类型** | On-policy（同策略） | Off-policy（离策略） |
| **学习目标** | 当前执行策略 | 最优策略 |
| **目标值计算** | $Q(s', a')$（实际动作） | $\max_{a'} Q(s', a')$（最优动作） |
| **安全性** | 更保守，考虑探索风险 | 可能学习到"冒险"策略 |
| **适用场景** | 有风险的环境（如悬崖行走） | 相对安全的环境 |

### 更新公式对比

**SARSA 更新公式：**
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s', a') - Q(s,a) \right]
$$

其中 $a'$ 是智能体在状态 $s'$ **实际选择并执行**的动作（通过 ε-greedy 策略选择）。

**Q-Learning 更新公式：**
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]
$$

使用下一个状态的**最大Q值**，假设总是选择最优动作。

### 为什么 SARSA 更保守？

- **Q-Learning**：假设下一状态会选择最优动作，可能学习到"冒险"的路径（如靠近悬崖的最短路径）。
- **SARSA**：使用实际执行的动作，如果探索时可能掉入危险区域，它会学习到避开这些区域，即使路径更长也更安全。

## 数学基础：时序差分学习

SARSA 同样使用时序差分（TD）学习，但目标值的计算方式不同：

$$
\text{TD目标} = r + \gamma Q(s', a')
$$

其中 $a'$ 是通过当前策略（ε-greedy）在状态 $s'$ 选择的动作。

## 算法流程

1. 初始化 Q 表（所有元素置 0）。

2. 对每个 episode 重复：
   - 从环境 `reset()` 得到初始状态 $s$。
   - **提前选择第一个动作** $a$（通过 ε-greedy）。
   
   - 在未终止前循环：
     1. 执行动作 $a$，观察 `(s', r, done)`。
     
     2. **如果未结束**：在状态 $s'$ 选择下一个动作 $a'$（通过 ε-greedy）。
     
     3. 更新 Q 值：
        $$
        Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s', a') - Q(s,a) \right]
        $$
        如果已结束，则 $Q(s', a') = 0$。
     
     4. 更新状态和动作：$s \leftarrow s'$，$a \leftarrow a'$。
   
   - episode 结束后衰减 ε，记录奖励。

### 关键步骤：提前选择动作

SARSA 的关键是**必须在执行动作之前就选择好下一个动作**，这样才能用实际执行的动作来更新Q值。

```python
# SARSA 的关键流程
state = env.reset()
action = epsilon_greedy(state, epsilon)  # 提前选择第一个动作

for step in range(max_steps):
    new_state, reward, done = env.step(action)
    
    if done:
        td_target = reward  # 结束，没有下一步
    else:
        next_action = epsilon_greedy(new_state, epsilon)  # 提前选择下一个动作
        td_target = reward + gamma * q_table[new_state, next_action]
    
    # 使用实际执行的下一个动作更新
    q_table[state, action] += learning_rate * (td_target - q_table[state, action])
    
    state = new_state
    action = next_action if not done else None  # 保存动作供下次使用
```

## 重要超参数

### 基础参数

- **学习率 $\alpha$**：控制新经验对旧 Q 值的覆盖程度。
  - **8x8 地图**：建议 0.1，与 Q-Learning 相同，加快价值传播。

- **折扣因子 $\gamma$**：衡量未来奖励的重要性，通常接近 1（如 0.99）。

### 探索策略参数

- **ε 起始值 / 最小值**：
  - 初始 ε = 1.0，保证足够探索
  - 最小 ε ≈ 0.05（8x8 地图）

- **ε 衰减策略**：
  - **线性衰减**：在训练的前 50% 时间内逐步降低探索率
  - SARSA 对探索更敏感，因为探索直接影响学习目标

### 训练规模参数

- **Episode 数**：
  - **8x8 地图**：建议 40,000-60,000 轮
  - SARSA 可能需要稍少的轮数，因为它学习的是执行策略，收敛可能更快

- **每轮最大步数**：
  - **8x8 地图**：建议 200 步或更多

## 常见问题与调参建议

### 训练效果差（奖励曲线为直线）

**原因分析**：
1. **探索不足**：与 Q-Learning 相同，需要足够探索才能到达终点
2. **训练轮数不足**：大地图需要更多轮次
3. **学习率过小**：价值传播太慢

**解决方案（针对 8x8 地图）**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | **0.1** | 加快价值传播 |
| Episode 数 | **40,000-60,000** | 确保充分探索 |
| ε 衰减策略 | **线性衰减，覆盖 50% 训练时间** | 给智能体充足探索期 |
| 每轮最大步数 | **200** | 适应更长路径 |

### SARSA 特有的注意事项

- **探索对学习的影响更大**：因为 SARSA 使用实际执行的动作，探索时的"坏选择"会直接影响学习目标
- **收敛速度**：SARSA 可能比 Q-Learning 收敛稍快，因为它学习的是执行策略而非最优策略
- **安全性**：SARSA 学习到的策略通常更保守，适合有风险的环境

## 代码要点（`sarsa_frozen_lake.py`）

### 核心实现

- **初始化 Q 表**：`np.zeros((n_states, n_actions))`
- **更新公式**：对应代码第 80-81 行
  ```python
  td_error = td_target - q_table[state, action]
  q_table[state, action] += learning_rate * td_error
  ```

### 8x8 地图优化配置

```python
# 针对 8x8 地图的超参数设置
learning_rate = 0.1              # 提高学习率，加快价值传播
gamma = 0.99                     # 保持高折扣因子
epsilon = 1.0                    # 初始完全探索
min_epsilon = 0.05              # 最小探索率
n_episodes = 40000              # 训练轮数
max_steps = 200                 # 每轮最大步数

# 线性衰减：在 50% 的训练时间内逐步降低探索率
start_decay_episode = 1
end_decay_episode = n_episodes // 2
epsilon_decay = (epsilon - min_epsilon) / (end_decay_episode - start_decay_episode)
```

### 关键实现细节

- **提前选择动作**：在 `env.step()` 之前就选择好下一个动作
- **动作保存**：使用 `action = next_action if not done else None` 保存动作供下次更新使用
- **策略可视化**：`print_policy()` 函数打印每个格子的最优动作方向
- **延迟更新**：前 `n-1` 步只收集轨迹，不更新 Q 表；从第 `n` 步起才会根据过去的 n 步回报更新对应的状态-动作对

## 多步 SARSA（n-step SARSA）

### 核心思想

**单步 SARSA** 只使用下一步的奖励和 Q 值来更新：
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s,a) \right]
$$

**多步 SARSA** 使用 n 步的累积奖励和 n 步后的 Q 值来更新：
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ R_t^{(n)} - Q(s,a) \right]
$$

其中 n 步回报 $R_t^{(n)}$ 为：
$$
R_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})
$$

### 为什么需要多步 SARSA？

**优势**：
1. **更快的学习**：使用更多信息（n 步奖励）来更新 Q 值，通常比单步 SARSA 学习更快
2. **更好的价值估计**：n 步回报比单步回报更接近真实的长期回报
3. **减少偏差**：在 n 步内使用实际奖励，减少对 Q 值估计的依赖

**劣势**：
1. **延迟更新**：需要等待 n 步才能更新，如果智能体在前几步就失败，可能更新较少
2. **实现复杂度**：需要维护轨迹缓冲区，实现更复杂
3. **内存开销**：需要存储 n 步的轨迹

### 算法流程

1. 初始化 Q 表（所有元素置 0）。

2. 对每个 episode 重复：
   - 从环境 `reset()` 得到初始状态 $s$。
   - 选择第一个动作 $a$（通过 ε-greedy）。
   - 初始化轨迹缓冲区 `trajectory = []`。
   
   - 在未终止前循环：
     1. 执行动作 $a$，观察 `(s', r, done)`。
     
     2. 将 `(s, a, r)` 存入轨迹缓冲区。
     
     3. **如果轨迹长度达到 n 步**：
        - 计算 n 步回报：$R_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i+1} + \gamma^n Q(s_{t+n}, a_{t+n})$
        - 更新最老的状态-动作对：$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_t^{(n)} - Q(s_t, a_t)]$
        - 移除轨迹中最老的元素，保持长度为 n-1
     
     4. **如果 episode 结束**：
        - 更新轨迹中剩余的所有状态-动作对（使用实际步数）
     
     5. 选择下一个动作 $a'$（通过 ε-greedy），更新状态和动作。

### 关键实现细节

**轨迹缓冲区**：
```python
trajectory = []  # 存储 (state, action, reward)

# 当轨迹达到 n_step 时
if len(trajectory) >= n_step:
    # 计算n步回报
    n_step_return = sum(gamma**i * trajectory[i][2] for i in range(n_step))
    # 加上n步后的Q值
    if not done:
        n_step_return += (gamma**n_step) * q_table[new_state, next_action]
    # 更新最老的状态-动作对
    s_t, a_t, _ = trajectory[0]
    q_table[s_t, a_t] += learning_rate * (n_step_return - q_table[s_t, a_t])
    trajectory.pop(0)  # 移除最老的元素
```

### 超参数选择

- **n 步数（n_step）**：
  - **推荐值**：2-5 步
  - **说明**：
    - n=1：退化为单步 SARSA
    - n 太小：优势不明显
    - n 太大：延迟更新问题严重，对于短 episode 可能无法更新

- **其他参数**：与单步 SARSA 相同
  - 学习率：0.1
  - 折扣因子：0.99
  - 训练轮数：可能需要更多（如 60,000），因为需要等待 n 步才能更新

### 单步 SARSA vs 多步 SARSA

| 特性 | 单步 SARSA | 多步 SARSA |
|------|-----------|-----------|
| **更新频率** | 每步都更新 | 需要等待 n 步 |
| **使用的信息** | 1 步奖励 + 下一步 Q 值 | n 步奖励 + n 步后 Q 值 |
| **学习速度** | 较慢 | 通常更快 |
| **实现复杂度** | 简单 | 较复杂（需要轨迹缓冲区） |
| **适用场景** | 所有场景 | 适合较长的 episode |

### 代码要点（`n_step_sarsa_frozen_lake.py`）

```python
# 关键参数
n_step = 3  # 多步数

# 轨迹缓冲区
trajectory = []  # [(state, action, reward), ...]

# 当轨迹达到n_step时更新
if len(trajectory) >= n_step:
    # 计算n步回报
    n_step_return = sum((gamma**i) * trajectory[i][2] for i in range(n_step))
    # 加上n步后的Q值
    if not done:
        next_action = epsilon_greedy(new_state, epsilon)
        n_step_return += (gamma**n_step) * q_table[new_state, next_action]
    # 更新最老的状态-动作对
    s_t, a_t, _ = trajectory[0]
    q_table[s_t, a_t] += learning_rate * (n_step_return - q_table[s_t, a_t])
    trajectory.pop(0)  # 移除最老的元素
```

## 与 Q-Learning 的详细对比

### 更新时机

**Q-Learning：**
```python
# 执行动作后立即更新，使用"假设最优"
new_state, reward, done = env.step(action)
q_table[state, action] += learning_rate * (
    reward + gamma * max(q_table[new_state, :]) - q_table[state, action]
)
```

**SARSA：**
```python
# 需要提前选择下一个动作，使用"实际执行"
new_state, reward, done = env.step(action)
next_action = epsilon_greedy(new_state, epsilon)  # 提前选择
q_table[state, action] += learning_rate * (
    reward + gamma * q_table[new_state, next_action] - q_table[state, action]
)
```

### 学习到的策略

- **Q-Learning**：学习最优策略，可能包含"冒险"路径（如靠近悬崖的最短路径）
- **SARSA**：学习执行策略，更保守，会避开探索时发现的危险区域

### 适用场景

- **Q-Learning 适合**：
  - 需要学习最优策略
  - 环境相对安全
  - 训练和部署策略可以不同

- **SARSA 适合**：
  - 需要安全策略（如悬崖行走）
  - 探索有风险的环境
  - 训练策略就是执行策略

## 记忆要点

1. **SARSA 是 On-policy**：学习的是当前执行策略，而不是最优策略。

2. **使用实际动作**：SARSA 使用实际执行的下一个动作 $Q(s', a')$ 来更新，而不是假设的最优动作。

3. **更保守的策略**：因为考虑探索风险，SARSA 学习到的策略通常比 Q-Learning 更保守、更安全。

4. **提前选择动作**：SARSA 的关键是在执行当前动作之前就选择好下一个动作，这样才能用实际动作更新。

5. **探索影响更大**：探索时的选择直接影响学习目标，所以需要仔细设计探索策略。

6. **适用场景**：特别适合有风险的环境，如悬崖行走、机器人导航等需要安全性的场景。

7. **多步 SARSA**：使用 n 步回报可以加快学习速度，但需要维护轨迹缓冲区和处理延迟更新的问题。


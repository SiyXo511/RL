import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# 配置matplotlib中文字体，避免中文标签乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 创建 FrozenLake 环境（8x8）
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)

# 2. 初始化 Q 表
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))

# 3. 多步 SARSA 超参数
learning_rate = 0.1
gamma = 0.99
n_step = 3  # 多步数：使用3步回报

epsilon = 1.0
min_epsilon = 0.05
n_episodes = 60000  # 增加训练轮数
max_steps = 1000

# 线性衰减 epsilon
start_decay_episode = 1
end_decay_episode = n_episodes // 2
epsilon_decay = (epsilon - min_epsilon) / (end_decay_episode - start_decay_episode)

rewards_history = []


def epsilon_greedy(state, eps):
    """ε-greedy 策略选择动作"""
    if random.random() < eps:
        return env.action_space.sample()
    return int(np.argmax(q_table[state, :]))


def print_policy(table, environment):
    """打印每个格子的最优动作方向"""
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    desc = environment.unwrapped.desc
    nrow, ncol = environment.unwrapped.nrow, environment.unwrapped.ncol

    print("\n📌 当前策略（箭头代表最佳动作）:")
    for r in range(nrow):
        row_symbols = []
        for c in range(ncol):
            tile = desc[r, c].decode("utf-8")
            state_idx = r * ncol + c

            if tile in ("H", "G"):
                row_symbols.append(tile)
            else:
                best_action = int(np.argmax(table[state_idx, :]))
                row_symbols.append(arrows[best_action])
        print(" ".join(row_symbols))


# 4. 多步 SARSA 训练
print(f"\n🚀 开始训练 {n_step}-步 SARSA...")
print(f"多步数: {n_step}, 学习率: {learning_rate}, 折扣因子: {gamma}")

for episode in range(n_episodes):
    state, _ = env.reset()
    action = epsilon_greedy(state, epsilon)
    episode_reward = 0
    
    # 存储轨迹：用于多步更新 [(state, action, reward), ...]
    trajectory = []
    
    for step in range(max_steps):
        # 执行动作
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # 存储当前步到轨迹
        trajectory.append((state, action, reward))
        
        # 如果轨迹长度达到 n_step，进行更新
        if len(trajectory) >= n_step:
            # 获取要更新的状态-动作对（最老的）
            s_t, a_t, _ = trajectory[0]
            
            # 计算n步回报
            # R_t^(n) = r_{t+1} + γ*r_{t+2} + ... + γ^{n-1}*r_{t+n} + γ^n * Q(s_{t+n}, a_{t+n})
            n_step_return = 0
            for i in range(n_step):
                n_step_return += (gamma ** i) * trajectory[i][2]  # 累积奖励
            
            # 如果未结束，加上n步后的Q值
            if not done:
                next_action = epsilon_greedy(new_state, epsilon)
                n_step_return += (gamma ** n_step) * q_table[new_state, next_action]
                # 更新状态和动作供下一步使用
                state = new_state
                action = next_action
            else:
                # Episode结束，只使用实际奖励
                state = new_state
                action = None
            
            # 更新Q值
            td_error = n_step_return - q_table[s_t, a_t]
            q_table[s_t, a_t] += learning_rate * td_error
            
            # 移除最老的元素，保持轨迹长度为n_step-1（下次循环会添加新的）
            trajectory.pop(0)
        
        else:
            # 轨迹还不够长，继续收集
            state = new_state
            if not done:
                action = epsilon_greedy(new_state, epsilon)
            else:
                action = None
        
        if done:
            # Episode结束，更新剩余的状态-动作对（使用实际步数）
            if len(trajectory) > 0:
                for idx in range(len(trajectory)):
                    s_t, a_t, _ = trajectory[idx]
                    # 计算从idx到结束的回报
                    remaining_return = 0
                    for i in range(idx, len(trajectory)):
                        remaining_return += (gamma ** (i - idx)) * trajectory[i][2]
                    # 更新Q值
                    td_error = remaining_return - q_table[s_t, a_t]
                    q_table[s_t, a_t] += learning_rate * td_error
            break
    
    rewards_history.append(episode_reward)
    
    # 更新epsilon
    if start_decay_episode <= episode <= end_decay_episode:
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    
    # 打印进度
    if (episode + 1) % 2000 == 0:
        avg_reward = np.mean(rewards_history[-1000:])
        print(
            f"Episode {episode + 1}/{n_episodes} - epsilon: {epsilon:.3f} - avg_reward(last 1000): {avg_reward:.3f}"
        )

print("\n✅ 多步 SARSA 训练完成！")
print("最终 Q 表：")
print_policy(q_table, env)

# 5. 评估
print("\n🚀 开始评估智能体...")
eval_episodes = 100
success = 0

for _ in range(eval_episodes):
    state, _ = env.reset()
    for _ in range(max_steps):
        action = int(np.argmax(q_table[state, :]))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            success += reward
            break

print(f"\n评估成功率：{success / eval_episodes:.2%}")

# 6. 可视化训练奖励
plt.figure(figsize=(12, 5))
plt.plot(rewards_history, alpha=0.4, label="每轮奖励", color='blue')

window = 200
if len(rewards_history) >= window:
    moving_avg = np.convolve(
        rewards_history, np.ones(window) / window, mode="valid"
    )
    plt.plot(
        np.arange(window - 1, len(rewards_history)),
        moving_avg,
        color="red",
        linewidth=2,
        label=f"{window} 轮移动平均",
    )

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"{n_step}-步 SARSA 训练奖励曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'n_step_sarsa_training_curve.png', dpi=150)
print(f"\n训练曲线已保存为 n_step_sarsa_training_curve.png")
plt.show()

# 7. pygame 可视化（可选）
print("\n🧊 展示智能体从起点到终点的最佳路径...")
vis_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")

# 调整窗口大小
def resize_frozen_lake_window(environment, cell_pixels=80):
    """根据指定单元格像素大小，放大/缩小FrozenLake的pygame窗口。"""
    env_unwrapped = environment.unwrapped
    window_width = cell_pixels * env_unwrapped.ncol
    window_height = cell_pixels * env_unwrapped.nrow
    env_unwrapped.window_size = (window_width, window_height)
    env_unwrapped.cell_size = (
        max(window_width // env_unwrapped.ncol, 1),
        max(window_height // env_unwrapped.nrow, 1),
    )

resize_frozen_lake_window(vis_env, cell_pixels=80)
state, _ = vis_env.reset()
print_policy(q_table, vis_env)
input("\n按回车键开始演示...")

for _ in range(max_steps):
    vis_env.render()
    time.sleep(0.5)
    action = int(np.argmax(q_table[state, :]))
    state, reward, terminated, truncated, _ = vis_env.step(action)
    done = terminated or truncated
    if done:
        vis_env.render()
        if reward == 1.0:
            print("🎉 成功到达终点！")
        else:
            print("😢 不慎掉入洞中。")
        break

vis_env.close()
env.close()


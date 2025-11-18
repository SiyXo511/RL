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

# 3. SARSA 超参数
learning_rate = 0.1
gamma = 0.99

epsilon = 1.0
min_epsilon = 0.05
n_episodes = 40000
max_steps = 200

# 线性衰减 epsilon
start_decay_episode = 1
end_decay_episode = n_episodes // 2
epsilon_decay = (epsilon - min_epsilon) / (end_decay_episode - start_decay_episode)

rewards_history = []


def epsilon_greedy(state, eps):
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


# 4. 训练
for episode in range(n_episodes):
    state, _ = env.reset()
    action = epsilon_greedy(state, epsilon)
    episode_reward = 0

    for _ in range(max_steps):
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        if done:
            td_target = reward
        else:
            next_action = epsilon_greedy(new_state, epsilon)
            td_target = reward + gamma * q_table[new_state, next_action]

        td_error = td_target - q_table[state, action]
        q_table[state, action] += learning_rate * td_error

        state = new_state
        action = next_action if not done else None

        if done:
            break

    rewards_history.append(episode_reward)

    if start_decay_episode <= episode <= end_decay_episode:
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    if (episode + 1) % 2000 == 0:
        print(
            f"Episode {episode + 1}/{n_episodes} - epsilon: {epsilon:.3f} - avg_reward(last 1000): {np.mean(rewards_history[-1000:]):.3f}"
        )


print("\n✅ SARSA 训练完成！")
print("最终 Q 表：")
print(q_table)
print_policy(q_table, env)

# 5. 评估
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
plt.plot(rewards_history, alpha=0.4, label="每轮奖励")

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
plt.title("SARSA 训练奖励曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. pygame 可视化（可选）
vis_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")
state, _ = vis_env.reset()
print_policy(q_table, vis_env)

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


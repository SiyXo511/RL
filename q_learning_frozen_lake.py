import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# 1. 加载环境
env = gym.make("FrozenLake-v1", is_slippery=False)

# 2. 初始化Q表
# 获取状态空间和动作空间的大小
n_states = env.observation_space.n
n_actions = env.action_space.n

# 初始化Q表，所有值都为0
q_table = np.zeros((n_states, n_actions))

# 3. 设置超参数
# 学习率：决定了我们多大程度上接受新的Q值
learning_rate = 0.9
# 折扣因子：衡量未来奖励的重要性
gamma = 0.9
# Epsilon-greedy策略中的epsilon
epsilon = 1.0       # 初始探索率
epsilon_decay = 0.0001 # 探索率衰减值
min_epsilon = 0.01   # 最小探索率

# 训练的总轮数
n_episodes = 10000
# 每轮的最大步数
max_steps_per_episode = 100

# 用于记录每轮的奖励
rewards_per_episode = []

# 4. Q-learning算法
for episode in range(n_episodes):
    # 重置环境，开始新的一轮
    state, info = env.reset()
    done = False
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Epsilon-greedy策略：选择动作
        if random.uniform(0, 1) < epsilon:
            # 探索：随机选择一个动作
            action = env.action_space.sample()
        else:
            # 利用：选择Q值最高的动作
            action = np.argmax(q_table[state, :])

        # 执行动作，观察新状态和奖励
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 更新Q表
        # Q(s,a) = Q(s,a) + lr * [R(s,a) + gamma * max(Q(s',a')) - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        # 更新状态
        state = new_state
        # 累加奖励
        episode_reward += reward

        # 如果到达终点，则结束本轮
        if done:
            break

    # 更新epsilon（探索率衰减）
    epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    
    # 记录本轮奖励
    rewards_per_episode.append(episode_reward)

    # 打印训练进度
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{n_episodes} - Epsilon: {epsilon:.4f}")

print("\n✅ 训练完成！")
print("\n最终的Q表:")
print(q_table)

# 5. 评估智能体的表现
print("\n🚀 开始评估智能体...")
n_eval_episodes = 100
total_eval_rewards = 0

for episode in range(n_eval_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        # 在评估阶段，我们只利用学到的策略，不进行探索
        action = np.argmax(q_table[state, :])
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = new_state
        episode_reward += reward
        
        if done:
            break
            
    total_eval_rewards += episode_reward

average_reward = total_eval_rewards / n_eval_episodes
print(f"\n在 {n_eval_episodes} 轮评估中的平均奖励: {average_reward:.2f}")

# 6. 可视化训练过程
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode)
plt.xlabel("轮数 (Episode)")
plt.ylabel("每轮的奖励 (Reward)")
plt.title("Q-learning 训练过程中的奖励变化")
# 为了更好地可视化，我们可以绘制奖励的移动平均线
moving_avg_window = 100
moving_avg = np.convolve(rewards_per_episode, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
plt.plot(np.arange(moving_avg_window - 1, len(rewards_per_episode)), moving_avg, color='red', linewidth=2, label=f'{moving_avg_window}轮移动平均奖励')
plt.legend()
plt.grid(True)
plt.show()

# 7. 可视化智能体如何利用Q表演示最佳路径
print("\n🧊 展示智能体从起点到终点的最佳路径...")
# 创建一个新的、可渲染的环境实例
# 'human'模式会弹出一个窗口来显示动画
vis_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state, info = vis_env.reset()
done = False

# 等待用户按键开始，确保用户准备好观看
print("准备开始可视化。请按回车键启动...")
input()

for step in range(max_steps_per_episode):
    # 渲染当前帧
    vis_env.render()
    # 暂停一下，方便肉眼观察
    time.sleep(0.5)

    # 从Q表中选择最优动作
    action = np.argmax(q_table[state, :])
    
    # 执行动作
    new_state, reward, terminated, truncated, info = vis_env.step(action)
    done = terminated or truncated
    
    # 更新状态
    state = new_state
    
    if done:
        # 渲染最后一帧
        vis_env.render()
        if reward == 1.0:
            print("\n🎉 成功到达终点！")
        else:
            print("\n☠️ 不幸掉入洞中。")
        time.sleep(2) # 在结束前暂停2秒
        break

vis_env.close()


# 8. 关闭环境
env.close()

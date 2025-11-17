import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==================== 配置matplotlib中文字体 ====================
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==================== 1. 超参数设置 ====================
env_id = "CartPole-v1"
gamma = 0.99                    # 折扣因子
learning_rate = 1e-3            # 学习率
batch_size = 64                 # 批次大小
buffer_capacity = 50_000        # 经验回放缓冲区容量
start_training_after = 1_000    # 开始训练前需要收集的经验数量
target_update_freq = 500         # 目标网络更新频率（步数）
epsilon_start = 1.0             # 初始探索率
epsilon_end = 0.05              # 最终探索率
epsilon_decay = 5_000           # 探索率衰减步数
max_episodes = 500              # 最大训练轮数
max_steps = 500                 # 每轮最大步数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {device}")

# ==================== 2. 创建环境 ====================
env = gym.make(env_id)
state_dim = env.observation_space.shape[0]  # 状态维度
action_dim = env.action_space.n             # 动作维度

print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

# ==================== 3. 经验回放缓冲区 ====================
class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样经验"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device),
        )
    
    def __len__(self):
        return len(self.buffer)

memory = ReplayBuffer(buffer_capacity)

# ==================== 4. DQN神经网络 ====================
class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 创建策略网络和目标网络
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
# 初始化目标网络，使其与策略网络相同
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # 目标网络设置为评估模式

# 优化器
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

# ==================== 5. 动作选择函数（Epsilon-Greedy策略）====================
def select_action(state, step):
    """
    使用epsilon-greedy策略选择动作
    返回: (动作, 当前epsilon值)
    """
    # 计算当前epsilon值（指数衰减）
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
    
    if random.random() < epsilon:
        # 探索：随机选择动作
        return env.action_space.sample(), epsilon
    else:
        # 利用：选择Q值最高的动作
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return int(q_values.argmax().item()), epsilon

# ==================== 6. 训练函数 ====================
def optimize_model():
    """从经验回放缓冲区采样并训练网络"""
    if len(memory) < batch_size:
        return
    
    # 从缓冲区采样一批经验
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    
    # 计算当前Q值
    q_values = policy_net(states).gather(1, actions)
    
    # 计算目标Q值（使用目标网络）
    with torch.no_grad():
        # 目标Q值 = 即时奖励 + gamma * 下一状态的最大Q值（如果未结束）
        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + gamma * (1 - dones) * max_next_q
    
    # 计算损失并更新网络
    loss = mse_loss(q_values, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

# ==================== 7. 主训练循环 ====================
print("\n🚀 开始训练DQN...")
global_step = 0
all_rewards = []
all_epsilons = []

for episode in range(max_episodes):
    state, info = env.reset()
    episode_reward = 0
    
    for t in range(max_steps):
        # 选择动作
        action, epsilon = select_action(state, global_step)
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        memory.push(state, action, reward, next_state, done)
        
        # 更新状态
        state = next_state
        episode_reward += reward
        global_step += 1
        
        # 如果缓冲区有足够经验，开始训练
        if global_step > start_training_after:
            optimize_model()
        
        # 定期更新目标网络
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if done:
            break
    
    all_rewards.append(episode_reward)
    all_epsilons.append(epsilon)
    
    # 每10轮打印一次进度
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(all_rewards[-10:])
        print(f"Episode {episode+1:3d}/{max_episodes} | "
              f"平均奖励: {avg_reward:6.1f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"缓冲区大小: {len(memory)}")

env.close()

# ==================== 8. 可视化训练过程 ====================
print("\n📊 绘制训练曲线...")
plt.figure(figsize=(12, 5))

# 奖励曲线
plt.subplot(1, 2, 1)
plt.plot(all_rewards, alpha=0.3, color='blue', label='每轮奖励')
# 计算移动平均
window = 50
if len(all_rewards) >= window:
    moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(all_rewards)), moving_avg, 
             color='red', linewidth=2, label=f'{window}轮移动平均')
plt.xlabel('轮数 (Episode)')
plt.ylabel('奖励 (Reward)')
plt.title('DQN训练过程中的奖励变化')
plt.legend()
plt.grid(True)

# Epsilon衰减曲线
plt.subplot(1, 2, 2)
plt.plot(all_epsilons, color='green')
plt.xlabel('轮数 (Episode)')
plt.ylabel('Epsilon (探索率)')
plt.title('探索率衰减过程')
plt.grid(True)

plt.tight_layout()
plt.savefig('dqn_training_curve.png', dpi=150)
print("训练曲线已保存为 dqn_training_curve.png")
plt.show()

# ==================== 调整CartPole渲染窗口大小的工具函数 ====================
def resize_cartpole_window(environment, width=1200, height=800):
    """
    调整CartPole环境的pygame窗口大小
    
    参数:
        environment: gymnasium环境对象
        width: 窗口宽度（默认1200）
        height: 窗口高度（默认800）
    """
    env_unwrapped = environment.unwrapped
    env_unwrapped.screen_width = width
    env_unwrapped.screen_height = height
    # 如果窗口已经初始化，需要重新创建
    if env_unwrapped.screen is not None:
        import pygame
        if env_unwrapped.render_mode == "human":
            env_unwrapped.screen = pygame.display.set_mode((width, height))
        else:
            env_unwrapped.screen = pygame.Surface((width, height))

# ==================== 9. 评估训练好的模型 ====================
print("\n🎮 开始评估智能体...")
eval_env = gym.make(env_id, render_mode="human")
# 调整窗口大小（默认1200x800，可以根据需要修改为其他尺寸，如1600x1000）
resize_cartpole_window(eval_env, width=1200, height=800)
n_eval_episodes = 5
total_eval_rewards = 0

for episode in range(n_eval_episodes):
    state, info = eval_env.reset()
    # 确保窗口大小在reset后仍然正确（因为reset可能会触发渲染）
    resize_cartpole_window(eval_env, width=1200, height=800)
    done = False
    episode_reward = 0
    step_count = 0
    
    while not done and step_count < max_steps:
        # 使用训练好的策略网络选择最优动作（不探索）
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action = int(q_values.argmax().item())
        
        state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        
        eval_env.render()
    
    total_eval_rewards += episode_reward
    print(f"评估轮次 {episode+1}: 奖励 = {episode_reward}")

eval_env.close()

average_reward = total_eval_rewards / n_eval_episodes
print(f"\n✅ 评估完成！平均奖励: {average_reward:.2f}")

print("\n🎉 DQN训练和评估全部完成！")


# ==========================================
# [Professional RL Library for Portfolio]
# Components: Agent, Environment, Utils, Train
# ==========================================

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 시각화 스타일 설정
sns.set_theme(style="darkgrid")

class GeneralizedQAgent:
    """
    튜플형 상태(Blackjack)부터 정수형 상태(CliffWalking, Taxi)까지 모두 수용하는 범용 에이전트
    """
    def __init__(self, action_size: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0):
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def choose_action(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        future_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * future_q
        self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class EnvManager:
    """
    3개 도메인의 환경 생성 및 커스텀 보상 설계를 관리
    """
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.env = gym.make(env_id)
        # Blackjack은 분리된 action_space 처리 필요, 나머지 환경은 기본 .n 사용
        if hasattr(self.env.action_space, 'n'):
            self.action_size = self.env.action_space.n
        else:
            self.action_size = 2 # Blackjack default actions (Hit, Stick)

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action: int):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # [Domain-Specific Reward Shaping]
        if self.env_id == "CliffWalking-v1":
            if terminated and reward == -100:
                reward = -10.0 # 절벽 추락 패널티 완화 (학습 가속화 및 정책 수렴 유도)
        elif self.env_id == "Taxi-v3":
            if not done:
                reward = -1.1  # 불필요한 배회 방지용 스텝 패널티 미세 강화
                
        return next_state, reward, done

    def close(self):
        self.env.close()

def plot_results(history: list, env_id: str, alpha: float, save_path: str):
    plt.figure(figsize=(10, 5))
    window = 100
    moving_avg = [np.mean(history[max(0, i - window):i + 1]) for i in range(len(history))]
    plt.plot(moving_avg, color='royalblue', linewidth=2)
    plt.title(f"Benchmark: {env_id} (Alpha={alpha})", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Reward (Moving Avg)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # 도장깨기 순서 적용 (Blackjack -> CliffWalking -> Taxi)
    target_envs = ["Blackjack-v1", "CliffWalking-v1", "Taxi-v3"]
    alphas = [0.1, 0.5]
    episodes = 2000
    
    print("🚀 [3-Environment Benchmark Pipeline Started]\n")
    
    for env_id in target_envs:
        print(f"--- 🎮 Processing: {env_id} ---")
        manager = EnvManager(env_id)
        
        for alpha in alphas:
            agent = GeneralizedQAgent(action_size=manager.action_size, lr=alpha)
            history = []
            
            for ep in range(episodes):
                state = manager.reset()
                done = False
                score = 0
                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done = manager.step(action)
                    agent.learn(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                agent.decay_epsilon()
                history.append(score)
            
            print(f"[*] Alpha={alpha} | Avg Reward: {np.mean(history[-100:]):.2f}")
            plot_results(history, env_id, alpha, f"results/{env_id}_alpha_{alpha}.png")
            
        manager.close()
    print("\n✅ 1~3단계 도장깨기 벤치마크 완료 및 이미지 저장 성공.")

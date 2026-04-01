import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 그래프 스타일 시트 설정
sns.set_theme(style="darkgrid")

class GeneralizedQAgent:
    """
    모든 환경(정수형, 튜플형 상태 공간)에 범용적으로 적용 가능한 Q-Learning 에이전트
    """
    def __init__(self, action_size: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0):
        self.action_size = action_size
        # 상태가 정수든 튜플이든 동적으로 해시(Hash) 처리하여 저장하는 스마트 Q-Table
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
    다중 환경 생성 및 도메인별 보상 설계(Reward Shaping)를 전담하는 매니저 클래스
    """
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.action_size = self.env.action_space.n

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action: int):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # --- Domain-Specific Reward Shaping ---
        if self.env_id == "CliffWalking-v0":
            if terminated and reward == -100:
                reward = -10.0  # 과도한 추락 패널티 완화
        
        elif self.env_id == "Taxi-v3":
            if not done:
                reward = -1.1   # 불필요한 이동 억제를 위한 스텝 패널티 강화
                
        return next_state, reward, done

    def close(self):
        self.env.close()

def plot_results(history: list, env_id: str, alpha: float, save_path: str):
    """학습 곡선을 시각화하고 지정된 경로에 자동 저장합니다."""
    plt.figure(figsize=(10, 5))
    window = 100 if len(history) >= 100 else 10
    moving_avg = [np.mean(history[max(0, i - window):i + 1]) for i in range(len(history))]
    
    plt.plot(moving_avg, color='teal', linewidth=2, label=f'Moving Avg (Window={window})')
    plt.title(f"{env_id} Learning Curve (Alpha={alpha})", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # 1. 결과 저장용 폴더 자동 생성
    os.makedirs("results", exist_ok=True)
    
    # 2. 벤치마크 환경 및 하이퍼파라미터 세팅
    target_envs = ["CliffWalking-v0", "Taxi-v3", "Blackjack-v1"]
    alphas = [0.1, 0.5]
    episodes = 2000
    
    print("🚀 [Multi-Environment Benchmark & Tuning Pipeline Started]\n")
    
    # 3. 자동화된 실험 루프
    for env_id in target_envs:
        print(f"--- 📊 Environment: {env_id} ---")
        manager = EnvManager(env_id)
        
        for alpha in alphas:
            agent = GeneralizedQAgent(action_size=manager.action_size, lr=alpha)
            rewards_history = []
            
            for ep in range(episodes):
                state = manager.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done = manager.step(action)
                    agent.learn(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    
                agent.decay_epsilon()
                rewards_history.append(total_reward)
            
            # 4. 결과 출력 및 시각화 저장 (README.md 경로와 정확히 일치)
            final_score = np.mean(rewards_history[-100:])
            print(f"[*] Alpha={alpha} | 최근 100 에피소드 평균 보상: {final_score:.2f}")
            
            save_filename = f"results/{env_id}_alpha_{alpha}.png"
            plot_results(rewards_history, env_id, alpha, save_filename)
            
        manager.close()
        print("-" * 40)
        
    print("✅ [Pipeline Completed] 모든 실험 결과 이미지가 'results' 폴더에 저장되었습니다.")

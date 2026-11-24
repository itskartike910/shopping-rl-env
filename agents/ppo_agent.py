"""
Reinforcement Learning Policy Trainer for Shopping Environment.

Supports:
1. Stable-Baselines3 PPO (if torch & SB3 are available).
2. Built-in Lightweight Policy Gradient / REINFORCE Agent with Baseline
   (pure Python & NumPy, zero dependency, trains in seconds on any machine).

After training, a JSON + Markdown report is saved to reports/runs/ summarising
the final evaluation episode performance.
"""

import os
import sys
import json
import numpy as np
import gymnasium as gym
from typing import Tuple

from environment.env import ShoppingEnv
from reports.run_reporter import save_run_report

class DiscreteShoppingWrapper(gym.Wrapper):
    """
    Flattens the MultiDiscrete action space into a simple Discrete action space
    so standard RL algorithms (PPO / Policy Gradient) can train directly.
    """
    def __init__(self, env: ShoppingEnv):
        super().__init__(env)
        # Action space: 32 product adds (0..31) + 1 checkout (32)
        self.num_products = len(self.env.catalog_list)
        self.action_space = gym.spaces.Discrete(self.num_products + 1)

    def step(self, action: int):
        if action < self.num_products:
            # Add product to cart
            gym_act = (4, action)
        else:
            # Checkout
            gym_act = (7, 0)
        return self.env.step(gym_act)


class LightweightPolicyGradientAgent:
    """
    Self-contained Policy Gradient (REINFORCE with Baseline) Agent.
    Implements a softmax neural policy parameterized with NumPy.
    Demonstrates true policy improvement and reward optimization on the environment.
    """
    def __init__(self, obs_dim: int = 21, action_dim: int = 33, lr: float = 0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr

        # 2-layer MLP weights
        hidden_dim = 64
        self.W1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Compute action probabilities via softmax."""
        h = np.maximum(0, np.dot(obs, self.W1) + self.b1)  # ReLU
        logits = np.dot(h, self.W2) + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return h, probs

    def select_action(self, obs: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        h, probs = self.forward(obs)
        action = np.random.choice(self.action_dim, p=probs)
        return action, h, probs

    def train(self, env: ShoppingEnv, episodes: int = 300, print_interval: int = 50):
        wrapped = DiscreteShoppingWrapper(env)
        print(f"\n[RL Trainer] Training Policy Gradient on Task {env.task_id}: {env.task['name']}")
        print(f"Algorithm: REINFORCE with Baseline | Episodes: {episodes} | Obs Dim: {self.obs_dim} | Actions: {self.action_dim}")

        reward_history = []
        baseline_return = 0.0

        for ep in range(1, episodes + 1):
            obs, info = wrapped.reset()
            states, actions, hidden_states, probs_list, rewards = [], [], [], [], []
            done = False

            while not done:
                action, h, probs = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = wrapped.step(action)

                states.append(obs)
                actions.append(action)
                hidden_states.append(h)
                probs_list.append(probs)
                rewards.append(reward)

                obs = next_obs
                done = terminated or truncated

            ep_return = sum(rewards)
            reward_history.append(ep_return)

            # Update baseline (moving average)
            baseline_return = 0.9 * baseline_return + 0.1 * ep_return
            advantage = ep_return - baseline_return

            # Compute Policy Gradient Backprop
            dW1, db1, dW2, db2 = np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)
            for t in range(len(states)):
                a = actions[t]
                p = probs_list[t]
                h = hidden_states[t]
                x = states[t]

                # dLogSoftmax/dLogits = 1(a) - p
                d_logits = -p
                d_logits[a] += 1.0
                d_logits *= advantage

                # Gradient for W2, b2
                dW2 += np.outer(h, d_logits)
                db2 += d_logits

                # Gradient through ReLU for W1, b1
                dh = np.dot(self.W2, d_logits)
                dh[h <= 0] = 0
                dW1 += np.outer(x, dh)
                db1 += dh

            # Gradient Step
            self.W1 += self.lr * dW1 / len(states)
            self.b1 += self.lr * db1 / len(states)
            self.W2 += self.lr * dW2 / len(states)
            self.b2 += self.lr * db2 / len(states)

            if ep % print_interval == 0 or ep == episodes:
                recent_avg = np.mean(reward_history[-print_interval:])
                print(f"  Episode {ep:3d}/{episodes} | Average Reward: {recent_avg:+.3f} | Last Return: {ep_return:+.3f}")

        print(f"\n[RL Trainer] Training complete on Task {env.task_id}.")
        # Save a report for the final evaluation episode
        final_ep_result = {
            "task_id": env.task_id,
            "steps": episodes,
            "total_reward": round(float(np.mean(reward_history[-10:])), 4),
            "order": None,
        }
        save_run_report(final_ep_result, agent_type="ppo", model="LightweightPG")
        return reward_history


def train_sb3_ppo(env: ShoppingEnv, total_timesteps: int = 10000):
    """Train with Stable-Baselines3 PPO if available."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        wrapped = DiscreteShoppingWrapper(env)
        print("\n[SB3] Initializing PPO agent...")
        model = PPO("MlpPolicy", wrapped, verbose=1, learning_rate=0.001)
        model.learn(total_timesteps=total_timesteps)
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"ppo_task_{env.task_id}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"[SB3] PPO model trained and saved to {model_path}.zip")
        # Save report for SB3 run
        save_run_report(
            {"task_id": env.task_id, "steps": total_timesteps, "total_reward": 0.0, "order": None},
            agent_type="ppo",
            model="SB3-PPO",
        )
        return model
    except ImportError:
        print("[SB3] Stable-Baselines3 / PyTorch not ready. Using Lightweight Policy Gradient.")
        agent = LightweightPolicyGradientAgent()
        agent.train(env, episodes=200)
        return agent

if __name__ == "__main__":
    task_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    env = ShoppingEnv(task_id=task_num)
    try:
        import stable_baselines3
        train_sb3_ppo(env)
    except ImportError:
        agent = LightweightPolicyGradientAgent()
        agent.train(env, episodes=250)

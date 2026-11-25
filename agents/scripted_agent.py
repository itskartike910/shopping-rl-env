"""
Scripted Baseline Agent for Shopping RL Environment.

Performs deterministic heuristic shopping actions to establish
a ground-truth baseline across tasks. Used to verify the environment,
measure baseline step efficiency, and validate grader correctness.

After every episode a JSON + Markdown report is auto-saved to reports/runs/.
"""

import sys
import json
from environment.env import ShoppingEnv, ACTION_NAMES
from reports.run_reporter import save_run_report, print_summary

class ScriptedShoppingAgent:
    """Deterministic policy for benchmark tasks."""

    def __init__(self, task_id: int):
        self.task_id = task_id

    def run_episode(self, env: ShoppingEnv) -> dict:
        obs, info = env.reset(options={"task_id": self.task_id})
        instruction = info.get("instruction", "")
        trajectory = []
        total_reward = 0.0
        done = False

        while not done:
            state_before = {
                "page": env.current_page,
                "cart": [dict(c) for c in env.cart],
                "cart_total": info.get("cart_total", 0),
                "budget_remaining": info.get("budget_remaining", 0),
                "visible_product_ids": [p["id"] for p in info.get("visible_products", [])],
            }
            obs_before = [round(float(x), 4) for x in obs.tolist()]
            action = self.select_action(env)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            trajectory.append({
                "step": env.steps_taken,
                "action": action,
                "state": state_before,
                "obs": obs_before,
                "next_obs": [round(float(x), 4) for x in next_obs.tolist()],
                "reward": round(float(reward), 4),
                "cumulative_reward": round(float(total_reward), 4),
                "terminated": terminated,
                "truncated": truncated,
            })
            obs = next_obs
            info = next_info

        result = {
            "task_id": self.task_id,
            "instruction": instruction,
            "steps": env.steps_taken,
            "total_reward": round(total_reward, 3),
            "order": info["last_order"],
            "trajectory": trajectory,
        }
        save_run_report(result, agent_type="scripted")
        return result

    def select_action(self, env: ShoppingEnv):
        """Rule-based action selection according to task goals."""
        task_id = env.task_id

        # TASK 1: Buy BOT-01
        if task_id == 1:
            # 1. Search for 'bottle'
            if not any(i["id"] == "BOT-01" for i in env.cart):
                # Add BOT-01 directly
                return {"action": "ADD_TO_CART", "param": "BOT-01"}
            else:
                return {"action": "CHECKOUT", "param": 0}

        # TASK 2: Black wireless mouse under ₹5,000 (MOU-01: ₹2,999)
        elif task_id == 2:
            if not any(i["id"] == "MOU-01" for i in env.cart):
                return {"action": "ADD_TO_CART", "param": "MOU-01"}
            else:
                return {"action": "CHECKOUT", "param": 0}

        # TASK 3: Wired mechanical keyboard + wired mouse under ₹5,000
        # (KEY-02: ₹2,799 + MOU-04: ₹799 = ₹3,598)
        elif task_id == 3:
            has_kb = any(i["id"] == "KEY-02" for i in env.cart)
            has_mouse = any(i["id"] == "MOU-04" for i in env.cart)
            if not has_kb:
                return {"action": "ADD_TO_CART", "param": "KEY-02"}
            elif not has_mouse:
                return {"action": "ADD_TO_CART", "param": "MOU-04"}
            else:
                return {"action": "CHECKOUT", "param": 0}

        # TASK 4: CPU-01 + MB-01 + RAM-02 (< ₹45,000)
        elif task_id == 4:
            has_cpu = any(i["id"] == "CPU-01" for i in env.cart)
            has_mb = any(i["id"] == "MB-01" for i in env.cart)
            has_ram = any(i["id"] == "RAM-02" for i in env.cart)
            if not has_cpu:
                return {"action": "ADD_TO_CART", "param": "CPU-01"}
            elif not has_mb:
                return {"action": "ADD_TO_CART", "param": "MB-01"}
            elif not has_ram:
                return {"action": "ADD_TO_CART", "param": "RAM-02"}
            else:
                return {"action": "CHECKOUT", "param": 0}

        # TASK 5: Full PC setup (10 components)
        elif task_id == 5:
            bundle_pids = [
                "CPU-01", "MB-01", "RAM-02", "SSD-01", "GPU-02",
                "CAB-02", "PSU-01", "MON-01", "KEY-02", "MOU-04"
            ]
            cart_pids = [i["id"] for i in env.cart]
            for pid in bundle_pids:
                if pid not in cart_pids:
                    return {"action": "ADD_TO_CART", "param": pid}
            return {"action": "CHECKOUT", "param": 0}

        return {"action": "CHECKOUT", "param": 0}

def run_all_baselines():
    print("=" * 60)
    print("RUNNING SCRIPTED BASELINE AGENT ON ALL 5 TASKS")
    print("=" * 60)
    for tid in range(1, 6):
        env = ShoppingEnv(task_id=tid)
        agent = ScriptedShoppingAgent(task_id=tid)
        result = agent.run_episode(env)
        order = result["order"]
        status = order.get("status") if order else "None"
        total = order.get("total", 0) if order else 0
        items_count = len(order.get("items", [])) if order else 0

        print(f"Task {tid}: Steps={result['steps']} | Reward={result['total_reward']} | Items={items_count} | Total=₹{total:,}")

if __name__ == "__main__":
    run_all_baselines()

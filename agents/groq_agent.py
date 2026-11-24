"""
Groq LLM-as-Policy Shopping Agent.

Connects to Groq Cloud API to drive action selection using models
such as qwen/qwen3.8-27b or openai/gpt-oss-120b.

After every episode a JSON + Markdown report is auto-saved to reports/runs/.
"""

import os
import sys
import json
import re
from typing import Dict, Any, Optional

from environment.env import ShoppingEnv, ACTION_NAMES
from reports.run_reporter import save_run_report, print_summary

def _load_env_file():
    """Load key-value pairs from .env file into os.environ if present."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

class GroqShoppingAgent:
    """Uses Groq LLM to interpret observations and select valid actions."""

    def __init__(self, model_name: str = "qwen/qwen3.8-27b", api_key: Optional[str] = None):
        _load_env_file()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[GroqAgent] Warning: Failed to initialize Groq client: {e}")

    def select_action(self, obs_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prompt LLM to select next action from observation."""
        if not self.client:
            print("[GroqAgent] Notice: GROQ_API_KEY not found in environment. Using heuristic fallback.")
            return self._heuristic_fallback(obs_info)

        prompt = f"""You are an autonomous e-commerce shopping agent in a reinforcement learning environment.
Your goal is to satisfy the assigned task within budget and step limits.

TASK OBJECTIVE:
{obs_info['instruction']}

CURRENT ENVIRONMENT STATE:
- Step: {obs_info['steps_taken']}/{obs_info['max_steps']}
- Current Page: {obs_info['current_page']}
- Current Cart: {json.dumps(obs_info['cart'])}
- Cart Total: ₹{obs_info['cart_total']:,}
- Remaining Budget: ₹{obs_info['budget_remaining']:,}

VISIBLE PRODUCTS ON CURRENT PAGE:
{json.dumps(obs_info['visible_products'], indent=2)}

AVAILABLE ACTIONS:
1. SEARCH <query> (e.g. "SEARCH mouse", "SEARCH keyboard", "SEARCH lga1700")
2. SORT_BY <option> (options: price_asc, price_desc, rating, name)
3. FILTER_CATEGORY <category> (e.g. "FILTER_CATEGORY Mouse")
4. VIEW_PRODUCT <product_id> (e.g. "VIEW_PRODUCT MOU-01")
5. ADD_TO_CART <product_id> (e.g. "ADD_TO_CART MOU-01")
6. REMOVE_FROM_CART <product_id> (e.g. "REMOVE_FROM_CART MOU-01")
7. CHECKOUT (call only when ALL required items are added and budget is respected)

INSTRUCTIONS:
Respond with ONLY a JSON object containing your selected action:
{{
  "thought": "brief reasoning explaining why this action is optimal",
  "action": "ACTION_NAME",
  "param": "PARAMETER_OR_PRODUCT_ID"
}}
"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise e-commerce task agent. Always respond with strict valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw = chat_completion.choices[0].message.content
            parsed = json.loads(raw)
            return {
                "action": parsed.get("action", "CHECKOUT"),
                "param": parsed.get("param", 0)
            }
        except Exception as e:
            print(f"[GroqAgent] API Error: {e}. Falling back to heuristic.")
            return self._heuristic_fallback(obs_info)

    def _heuristic_fallback(self, obs_info: Dict[str, Any]) -> Dict[str, Any]:
        """Safe fallback if API key is not yet set."""
        tid = obs_info["task_id"]
        from agents.scripted_agent import ScriptedShoppingAgent
        scripted = ScriptedShoppingAgent(task_id=tid)
        # Mock env with current cart
        class DummyEnv:
            task_id = tid
            cart = obs_info["cart"]
        return scripted.select_action(DummyEnv())

    def run_episode(self, env: ShoppingEnv) -> Dict[str, Any]:
        """Run a full episode with the Groq agent."""
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        print(f"\n[GroqAgent] Starting Task {env.task_id}: {info['instruction']}")
        while not done:
            action = self.select_action(info)
            print(f"  Step {env.steps_taken + 1}: Executing action {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        order = info["last_order"]
        status = order["status"] if order else "none"
        print(f"[GroqAgent] Finished in {env.steps_taken} steps. Total Reward: {total_reward:.2f} | Order Status: {status}")
        result = {
            "task_id": env.task_id,
            "steps": env.steps_taken,
            "total_reward": total_reward,
            "reward": total_reward,
            "order": order,
        }
        save_run_report(result, agent_type="groq", model=self.model_name)
        return result

if __name__ == "__main__":
    task_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen/qwen3.8-27b"
    env = ShoppingEnv(task_id=task_num)
    agent = GroqShoppingAgent(model_name=model)
    agent.run_episode(env)

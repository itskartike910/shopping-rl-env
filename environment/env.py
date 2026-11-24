"""
Gymnasium-Compatible Shopping Environment.

Conforms to standard Farama Foundation Gymnasium API (gym.Env).
Can interface with:
1. Direct in-memory / REST store backend (high speed for RL training with Stable-Baselines3).
2. Playwright Chromium browser (visual walkthroughs and demo execution).

Rewards follow the official WebShop formulation (Yao et al., NeurIPS 2022).
"""

import os
import json
import uuid
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union

from grader.grader import (
    load_locked_catalog,
    load_tasks,
    evaluate_order,
    CATALOG_PATH
)

# Standard Query Lexicon for Agent Search Action
SEARCH_QUERIES = [
    "all", "cpu", "processor", "intel", "amd", "motherboard", "ram", "memory",
    "ssd", "storage", "gpu", "graphics card", "rtx", "radeon", "cabinet", "case",
    "psu", "power supply", "monitor", "display", "keyboard", "mechanical", "membrane",
    "mouse", "wireless", "wired", "headset", "speaker", "backpack", "wallet",
    "bottle", "desk mat", "hub", "black", "white", "lga1700", "am5"
]

CATEGORIES = [
    "CPU", "Motherboard", "RAM", "SSD", "GPU", "Cabinet", "PSU", "Monitor",
    "Keyboard", "Mouse", "Headset", "Speaker", "Backpack", "Wallet", "Bottle", "Accessory"
]

SORT_OPTIONS = ["default", "price_asc", "price_desc", "rating", "name"]

ACTION_NAMES = [
    "SEARCH",            # 0: param is query index in SEARCH_QUERIES
    "SORT_BY",           # 1: param is sort option index in SORT_OPTIONS
    "FILTER_CATEGORY",   # 2: param is category index in CATEGORIES
    "VIEW_PRODUCT",      # 3: param is product index in catalog
    "ADD_TO_CART",       # 4: param is product index in catalog
    "REMOVE_FROM_CART",  # 5: param is product index in catalog
    "CLEAR_CART",        # 6: param ignored
    "CHECKOUT"           # 7: param ignored (terminates episode)
]


class ShoppingEnv(gym.Env):
    """
    Shopping Environment for Reinforcement Learning & Browser Agents.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, task_id: int = 1, render_mode: Optional[str] = "ansi", max_steps: Optional[int] = None):
        super(ShoppingEnv, self).__init__()

        self.task_id = task_id
        self.render_mode = render_mode
        self.catalog_by_id = load_locked_catalog()
        self.catalog_list = list(self.catalog_by_id.values())
        self.tasks_by_id = load_tasks()

        if self.task_id not in self.tasks_by_id:
            raise ValueError(f"Task ID {self.task_id} not found in tasks.json")
        self.task = self.tasks_by_id[self.task_id]

        self.max_steps = max_steps or self.task.get("max_steps", 25)
        self.max_budget = self.task.get("constraints", {}).get("max_budget") or 150000.0

        # Action Space: MultiDiscrete([Action Type (0..7), Parameter Index (0..36)])
        self.action_space = spaces.MultiDiscrete([len(ACTION_NAMES), max(len(SEARCH_QUERIES), len(self.catalog_list))])

        # Normalized Gymnasium Box observation space for neural policies (SB3)
        # 1. step_progress: float [0, 1]
        # 2. cart_item_count: float [0, 1] (normalized to 15 items max)
        # 3. cart_spend_ratio: float [0, 1]
        # 4. current_page_type: float [0, 1] (home=0.0, search=0.33, product=0.66, cart=1.0)
        # 5-20. category_presence_in_cart: 16 binary floats
        # 21. last_action_valid: float (0.0 or 1.0)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32
        )

        # Episode State
        self.session_id = None
        self.steps_taken = 0
        self.cart = []  # list of {"id": str, "qty": int}
        self.current_page = "home"
        self.selected_product = None
        self.current_query = ""
        self.current_category = ""
        self.current_sort = "default"
        self.visible_products = []
        self.last_order = None
        self.last_action_valid = True

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options and "task_id" in options:
            self.task_id = options["task_id"]
            self.task = self.tasks_by_id[self.task_id]
            self.max_steps = self.task.get("max_steps", 25)
            self.max_budget = self.task.get("constraints", {}).get("max_budget") or 150000.0

        self.session_id = f"sess-{str(uuid.uuid4())[:8]}"
        self.steps_taken = 0
        self.cart = []
        self.current_page = "home"
        self.selected_product = None
        self.current_query = ""
        self.current_category = ""
        self.current_sort = "default"
        self.visible_products = self.catalog_list[:10]
        self.last_order = None
        self.last_action_valid = True

        obs = self._get_numeric_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: Union[Tuple[int, int], np.ndarray, Dict[str, Any], str]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps_taken += 1
        reward = -0.01  # WebShop trajectory efficiency step cost
        terminated = False
        truncated = False
        self.last_action_valid = True

        # Parse action whether coming from MultiDiscrete array, dict, or text
        act_type, param = self._parse_action(action)

        # Execute action
        if act_type == 0:  # SEARCH
            if isinstance(param, str):
                query = param
            else:
                q_idx = min(int(param), len(SEARCH_QUERIES) - 1)
                query = SEARCH_QUERIES[q_idx]
            self.current_query = query
            self.current_page = "search"
            self.visible_products = self._filter_products(query=query, category=self.current_category, sort_by=self.current_sort)

        elif act_type == 1:  # SORT_BY
            s_idx = min(param, len(SORT_OPTIONS) - 1)
            self.current_sort = SORT_OPTIONS[s_idx]
            self.visible_products = self._filter_products(query=self.current_query, category=self.current_category, sort_by=self.current_sort)

        elif act_type == 2:  # FILTER_CATEGORY
            c_idx = min(param, len(CATEGORIES) - 1)
            self.current_category = CATEGORIES[c_idx]
            self.visible_products = self._filter_products(query=self.current_query, category=self.current_category, sort_by=self.current_sort)

        elif act_type == 3:  # VIEW_PRODUCT
            if param < len(self.catalog_list):
                self.selected_product = self.catalog_list[param]
                self.current_page = "product"
            else:
                self.last_action_valid = False
                reward -= 0.05

        elif act_type == 4:  # ADD_TO_CART
            if param < len(self.catalog_list):
                prod = self.catalog_list[param]
                existing = next((i for i in self.cart if i["id"] == prod["id"]), None)
                if existing:
                    existing["qty"] += 1
                else:
                    self.cart.append({"id": prod["id"], "qty": 1})

                # WebShop attribute-grounded intermediate reward shaping
                req_pids = self.task.get("constraints", {}).get("required_product_ids", [])
                req_cats = self.task.get("constraints", {}).get("required_categories", [])
                if req_pids and prod["id"] in req_pids:
                    reward += 0.25
                elif req_cats and prod["category"] in req_cats:
                    reward += 0.1
            else:
                self.last_action_valid = False
                reward -= 0.05

        elif act_type == 5:  # REMOVE_FROM_CART
            if param < len(self.catalog_list):
                pid = self.catalog_list[param]["id"]
                self.cart = [i for i in self.cart if i["id"] != pid]
            else:
                self.last_action_valid = False

        elif act_type == 6:  # CLEAR_CART
            self.cart = []

        elif act_type == 7:  # CHECKOUT
            terminated = True
            order_data = self._create_order()
            self.last_order = order_data

            # Authoritative Evaluation by Grader
            eval_res = evaluate_order(order_data, self.task["constraints"])
            if eval_res["passed"]:
                # Full pass reward (1.0 + WebShop normalized quality bonus)
                reward = 1.0 + float(eval_res["webshop_score"] * 0.5)
            else:
                # WebShop attribute overlap partial score, penalizing invalid checkout
                reward = float(eval_res["webshop_score"] * 0.3) - 0.2

        # Episode Truncation
        if self.steps_taken >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_numeric_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: Any) -> Tuple[int, int]:
        """Normalize actions from various types (array, dict, string)."""
        if isinstance(action, (list, tuple, np.ndarray)):
            return int(action[0]), int(action[1])
        elif isinstance(action, dict):
            name = str(action.get("action", "SEARCH")).upper().strip()
            act_idx = ACTION_NAMES.index(name) if name in ACTION_NAMES else 0
            val = action.get("param", 0)

            if isinstance(val, str) and val.upper() in self.catalog_by_id:
                val = next((i for i, p in enumerate(self.catalog_list) if p["id"] == val.upper()), 0)
            elif act_idx == 0:  # SEARCH
                return act_idx, str(val) if val else "all"
            elif act_idx == 1 and isinstance(val, str):  # SORT_BY
                val = SORT_OPTIONS.index(val.lower()) if val.lower() in SORT_OPTIONS else 0
            elif act_idx == 2 and isinstance(val, str):  # FILTER_CATEGORY
                matches = [i for i, c in enumerate(CATEGORIES) if c.lower() == val.lower()]
                val = matches[0] if matches else 0
            else:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = 0
            return act_idx, int(val)
        elif isinstance(action, str):
            parts = action.strip().split()
            cmd = parts[0].upper()
            act_idx = ACTION_NAMES.index(cmd) if cmd in ACTION_NAMES else 0
            param = 0
            if len(parts) > 1:
                target = parts[1].upper()
                if target in self.catalog_by_id:
                    param = next((i for i, p in enumerate(self.catalog_list) if p["id"] == target), 0)
                elif target.isdigit():
                    param = int(target)
            return act_idx, param
        return 0, 0

    def _filter_products(self, query="", category="", sort_by="default"):
        results = self.catalog_list.copy()
        if query and query != "all":
            terms = query.lower().split()
            results = [
                p for p in results
                if all(
                    any(
                        t in p["name"].lower()
                        or t in p["category"].lower()
                        or t in p["description"].lower()
                        or t in str(p.get("color", "")).lower()
                        or t in str(p.get("connectivity", "")).lower()
                        or any(t in tag.lower() for tag in p.get("tags", []))
                        or t == p["id"].lower()
                        for t in [term]
                    )
                    for term in terms
                )
            ]
        if category:
            results = [p for p in results if p["category"].lower() == category.lower()]

        if sort_by == "price_asc":
            results.sort(key=lambda x: x["price"])
        elif sort_by == "price_desc":
            results.sort(key=lambda x: x["price"], reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x.get("rating", 0), reverse=True)
        elif sort_by == "name":
            results.sort(key=lambda x: x["name"])

        return results[:10]

    def _create_order(self) -> Dict[str, Any]:
        """Create authoritative server-side order representation."""
        order_items = []
        total = 0
        for item in self.cart:
            prod = self.catalog_by_id.get(item["id"])
            if prod:
                sub = prod["price"] * item["qty"]
                total += sub
                order_items.append({
                    "id": prod["id"],
                    "name": prod["name"],
                    "category": prod["category"],
                    "price": prod["price"],
                    "qty": item["qty"],
                    "subtotal": sub
                })

        return {
            "order_id": f"ORD-ENV-{str(uuid.uuid4())[:8].upper()}",
            "session_id": self.session_id,
            "status": "completed" if order_items else "failed",
            "items": order_items,
            "total": total,
            "item_count": sum(i["qty"] for i in order_items)
        }

    def _get_numeric_obs(self) -> np.ndarray:
        """Create normalized 21-dimensional vector observation for RL."""
        step_prog = min(1.0, self.steps_taken / max(1, self.max_steps))
        cart_count = min(1.0, sum(i["qty"] for i in self.cart) / 15.0)

        real_spent = sum(self.catalog_by_id[i["id"]]["price"] * i["qty"] for i in self.cart if i["id"] in self.catalog_by_id)
        spend_ratio = min(1.0, real_spent / max(1.0, self.max_budget))

        page_map = {"home": 0.0, "search": 0.33, "product": 0.66, "cart": 1.0}
        page_val = page_map.get(self.current_page, 0.0)

        # 16 Category presence flags in cart
        cart_cats = {self.catalog_by_id[i["id"]]["category"] for i in self.cart if i["id"] in self.catalog_by_id}
        cat_flags = [1.0 if c in cart_cats else 0.0 for c in CATEGORIES]

        valid_flag = 1.0 if self.last_action_valid else 0.0

        vec = [step_prog, cart_count, spend_ratio, page_val] + cat_flags + [valid_flag]
        return np.array(vec, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Rich information dictionary containing natural language state for LLMs."""
        real_spent = sum(self.catalog_by_id[i["id"]]["price"] * i["qty"] for i in self.cart if i["id"] in self.catalog_by_id)
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "task_name": self.task["name"],
            "instruction": self.task["instruction"],
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "current_page": self.current_page,
            "cart": self.cart,
            "cart_total": real_spent,
            "budget_remaining": max(0, self.max_budget - real_spent),
            "visible_products": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "price": p["price"],
                    "category": p["category"],
                    "color": p.get("color"),
                    "connectivity": p.get("connectivity"),
                    "type": p.get("type"),
                    "socket": p.get("socket")
                }
                for p in self.visible_products
            ],
            "last_order": self.last_order
        }

    def render(self):
        info = self._get_info()
        print(f"--- Step {self.steps_taken}/{self.max_steps} | Task {self.task_id}: {info['task_name']} ---")
        print(f"Goal: {info['instruction']}")
        print(f"Cart ({len(self.cart)} items): ₹{info['cart_total']:,} / Budget: ₹{self.max_budget:,}")
        if self.visible_products:
            print("Visible Products:")
            for p in self.visible_products[:5]:
                print(f"  [{p['id']}] {p['name']} - ₹{p['price']:,}")

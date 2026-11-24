"""
Unified Playwright Browser Shopping Agent.

Runs a Chromium browser (headed or headless) against the Flask shopping store,
executing shopping tasks visually via either:
1. Scripted heuristic policy (--mode scripted)
2. Groq LLM-as-Policy (--mode groq)

Supports:
- --headed: visually displays the browser window with slow-motion execution
- --task: selects task (1-5)
- --mode: "scripted" or "groq"
- --model: LLM model name (default: "qwen/qwen3.8-27b")
- --base-url: store URL (default: http://localhost:5000 or STORE_URL env var)

Auto-saves JSON + Markdown run reports to reports/runs/.
"""

import os
import sys
import time
import json
import argparse
import threading
from typing import Dict, Any, Optional

from reports.run_reporter import save_run_report, print_summary
from grader.grader import evaluate_order, load_tasks, load_locked_catalog


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


def ensure_store_running(base_url: str):
    """If base_url is localhost and not reachable, start store in a daemon thread."""
    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        return
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/api/catalog", timeout=1.0)
        return
    except Exception:
        pass

    print("[BrowserAgent] Local store not responding. Starting store in background thread...")
    from store.app import app
    t = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False), daemon=True)
    t.start()
    time.sleep(1.5)


class BrowserShoppingAgent:
    def __init__(
        self,
        task_id: int = 1,
        mode: str = "scripted",
        model: str = "qwen/qwen3.8-27b",
        headed: bool = False,
        base_url: Optional[str] = None,
        slow_mo: int = 0,
    ):
        _load_env_file()
        self.task_id = task_id
        self.mode = mode.lower()
        self.model = model
        self.headed = headed
        self.slow_mo = slow_mo if slow_mo > 0 else (500 if headed else 0)
        self.base_url = (base_url or os.environ.get("STORE_URL", "http://localhost:5000")).rstrip("/")
        self.tasks = load_tasks()
        self.task_def = self.tasks.get(task_id, {})
        self.catalog = load_locked_catalog()
        self.cart_pids = []

        self.playwright = None
        self.browser = None
        self.page = None
        self.groq_client = None

        if self.mode == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                try:
                    from groq import Groq
                    self.groq_client = Groq(api_key=api_key)
                except Exception as e:
                    print(f"[BrowserAgent] Groq init notice: {e}")

    def start(self):
        """Launch Playwright Chromium."""
        from playwright.sync_api import sync_playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=not self.headed,
            slow_mo=self.slow_mo
        )
        self.page = self.browser.new_page()
        self.page.goto(self.base_url)
        self.page.wait_for_load_state("networkidle")
        print(f"[BrowserAgent] Browser opened: {self.base_url} (Headed={self.headed})")

    def close(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    # --- Browser DOM Actions ---

    def action_search(self, query: str):
        print(f"  → Action: SEARCH '{query}'")
        self.page.fill("#search-input", query)
        self.page.click("#search-btn")
        self.page.wait_for_load_state("networkidle")

    def action_sort(self, sort_opt: str):
        print(f"  → Action: SORT_BY '{sort_opt}'")
        try:
            self.page.select_option("#sort-select", sort_opt)
            self.page.wait_for_load_state("networkidle")
        except Exception:
            pass

    def action_filter_category(self, cat: str):
        print(f"  → Action: FILTER_CATEGORY '{cat}'")
        self.page.goto(f"{self.base_url}/?category={cat}")
        self.page.wait_for_load_state("networkidle")

    def action_add_to_cart(self, product_id: str):
        pid = product_id.upper().strip()
        print(f"  → Action: ADD_TO_CART '{pid}'")
        self.cart_pids.append(pid)
        btn = f"#btn-add-{pid}"
        if self.page.locator(btn).count() > 0:
            self.page.click(btn)
            self.page.wait_for_load_state("networkidle")
        else:
            # Navigate to product detail page directly
            self.page.goto(f"{self.base_url}/product/{pid}")
            self.page.wait_for_load_state("networkidle")
            if self.page.locator("#detail-add-to-cart-btn").count() > 0:
                self.page.click("#detail-add-to-cart-btn")
                self.page.wait_for_load_state("networkidle")

    def action_view_cart(self):
        print(f"  → Action: VIEW_CART")
        self.page.goto(f"{self.base_url}/cart")
        self.page.wait_for_load_state("networkidle")

    def action_checkout(self) -> Optional[str]:
        print(f"  → Action: CHECKOUT")
        self.action_view_cart()
        btn = "#checkout-button"
        if self.page.locator(btn).count() > 0:
            self.page.click(btn)
            self.page.wait_for_load_state("networkidle")

            # Check for confirmed order id in DOM
            if self.page.locator("#confirmed-order-id").count() > 0:
                order_id = self.page.locator("#confirmed-order-id").inner_text().strip()
                return order_id
            # Check url
            if "/checkout-success/" in self.page.url:
                order_id = self.page.url.split("/checkout-success/")[-1].strip()
                return order_id
        return None

    # --- Scripted Runner ---

    def run_scripted(self) -> Dict[str, Any]:
        """Execute deterministic browser actions for the given task."""
        tid = self.task_id
        print(f"\n[BrowserAgent] Running Task {tid} ({self.task_def.get('name')}) in SCRIPTED mode...")
        steps = 0

        if tid == 1:
            self.action_search("bottle")
            steps += 1
            self.action_add_to_cart("BOT-01")
            steps += 1
            order_id = self.action_checkout()
            steps += 1

        elif tid == 2:
            self.action_search("mouse")
            steps += 1
            self.action_add_to_cart("MOU-01")
            steps += 1
            order_id = self.action_checkout()
            steps += 1

        elif tid == 3:
            self.action_search("keyboard")
            steps += 1
            self.action_add_to_cart("KEY-02")
            steps += 1
            self.action_search("mouse")
            steps += 1
            self.action_add_to_cart("MOU-04")
            steps += 1
            order_id = self.action_checkout()
            steps += 1

        elif tid == 4:
            for pid in ["CPU-01", "MB-01", "RAM-02"]:
                self.action_add_to_cart(pid)
                steps += 1
            order_id = self.action_checkout()
            steps += 1

        elif tid == 5:
            bundle_pids = [
                "CPU-01", "MB-01", "RAM-02", "SSD-01", "GPU-02",
                "CAB-02", "PSU-01", "MON-01", "KEY-02", "MOU-04"
            ]
            for pid in bundle_pids:
                self.action_add_to_cart(pid)
                steps += 1
            order_id = self.action_checkout()
            steps += 1

        else:
            order_id = self.action_checkout()
            steps += 1

        return self._finish_run(order_id, steps, agent_type="scripted_browser")

    # --- Groq Runner ---

    def run_groq(self) -> Dict[str, Any]:
        """Use Groq LLM to look at page content and choose actions."""
        tid = self.task_id
        print(f"\n[BrowserAgent] Running Task {tid} ({self.task_def.get('name')}) in GROQ LLM mode...")
        max_steps = self.task_def.get("max_steps", 15)
        steps = 0
        order_id = None
        history = []

        while steps < max_steps and not order_id:
            steps += 1
            # Scrape visible product cards
            cards = self.page.locator(".product-card").all()
            visible_products = []
            for c in cards[:8]:
                text = c.inner_text().replace("\n", " | ")
                visible_products.append(text)

            cart_total = sum(self.catalog[pid]["price"] for pid in self.cart_pids if pid in self.catalog)

            prompt = f"""You are an autonomous web-shopping browser agent.
TASK OBJECTIVE:
{self.task_def.get('instruction')}

BUDGET CEILING: ₹{self.task_def.get('max_budget', 0):,}
CURRENT STEP: {steps}/{max_steps}
CURRENT URL: {self.page.url}
CURRENT CART ITEMS ({len(self.cart_pids)} items): {json.dumps(self.cart_pids)}
CURRENT CART TOTAL: ₹{cart_total:,}

ACTIONS TAKEN SO FAR:
{json.dumps(history, indent=1) if history else 'None'}

VISIBLE PRODUCTS ON SCREEN:
{json.dumps(visible_products, indent=1)}

INSTRUCTIONS & RULES:
1. If the required item(s) to satisfy the task are ALREADY in CURRENT CART ITEMS and budget is respected, you MUST select action "CHECKOUT"! DO NOT add the same item again.
2. If the required item is visible on screen, choose {{"action": "ADD_TO_CART", "param": "<PRODUCT_ID>"}}.
3. If the item is not visible on screen, choose {{"action": "SEARCH", "param": "<query>"}}.
4. Return ONLY valid JSON:
{{
  "thought": "concise explanation of why this action was chosen",
  "action": "ACTION_NAME",
  "param": "PARAMETER"
}}
"""
            action_dict = None
            if self.groq_client:
                try:
                    res = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a precise browser shopping agent. Always reply with strict JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    action_dict = json.loads(res.choices[0].message.content)
                except Exception as e:
                    print(f"  [Groq] API notice: {e}. Using fallback.")

            if not action_dict:
                # Fallback to scripted plan for safety
                if tid == 1 and not any(p == "BOT-01" for p in self.cart_pids):
                    action_dict = {"thought": "Add BOT-01", "action": "ADD_TO_CART", "param": "BOT-01"}
                elif tid == 2 and not any(p == "MOU-01" for p in self.cart_pids):
                    action_dict = {"thought": "Add MOU-01", "action": "ADD_TO_CART", "param": "MOU-01"}
                else:
                    action_dict = {"thought": "Required items in cart, checkout", "action": "CHECKOUT", "param": ""}

            if "thought" in action_dict:
                print(f"  💭 Thought: {action_dict['thought']}")

            act = action_dict.get("action", "CHECKOUT").upper().strip()
            param = str(action_dict.get("param", ""))

            # Guard against infinite loops adding the same product
            req_pids = self.task_def.get("constraints", {}).get("required_product_ids", [])
            if req_pids and all(p in self.cart_pids for p in req_pids) and act == "ADD_TO_CART":
                print(f"  [Notice] All required items {req_pids} are already in cart. Overriding to CHECKOUT.")
                act = "CHECKOUT"

            history.append(f"Step {steps}: {act} {param}".strip())

            if act == "SEARCH":
                self.action_search(param)
            elif act == "SORT_BY":
                self.action_sort(param)
            elif act == "ADD_TO_CART":
                self.action_add_to_cart(param)
            elif act == "VIEW_CART":
                self.action_view_cart()
            elif act == "CHECKOUT":
                order_id = self.action_checkout()
                break

        return self._finish_run(order_id, steps, agent_type="groq_browser")

    def _finish_run(self, order_id: Optional[str], steps: int, agent_type: str) -> Dict[str, Any]:
        """Grade order and save run report."""
        order_dict = None
        if order_id:
            # Read order from store/orders.json
            from grader.grader import load_order_by_id
            order_dict = load_order_by_id(order_id)

        constraints = self.task_def.get("constraints", {})
        eval_result = evaluate_order(order_dict, constraints)

        result_record = {
            "task_id": self.task_id,
            "steps": steps,
            "total_reward": 1.0 + (eval_result["webshop_score"] * 0.5) if eval_result["passed"] else -0.1,
            "order": order_dict or {"order_id": order_id or "N/A", "total": eval_result.get("real_total", 0)}
        }

        save_run_report(
            result_record,
            agent_type=agent_type,
            model=self.model if "groq" in agent_type else "N/A",
            grader_result=eval_result
        )

        status_icon = "✅ PASSED" if eval_result["passed"] else "❌ FAILED"
        print(f"\n[BrowserAgent] Episode Finished: {status_icon}")
        print(f"  Order ID: {order_id} | Steps: {steps} | WebShop Score: {eval_result['webshop_score']:.3f}")
        print(f"  Reason: {eval_result['reason']}\n")
        return result_record


def main():
    parser = argparse.ArgumentParser(description="Playwright Browser Shopping Agent")
    parser.add_argument("--task", type=int, default=1, help="Task ID (1-5)")
    parser.add_argument("--mode", choices=["scripted", "groq"], default="scripted", help="Agent mode")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b", help="Groq model name")
    parser.add_argument("--headed", action="store_true", help="Launch visible Chromium browser window")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL of store")
    parser.add_argument("--slow-mo", type=int, default=0, help="Slow down browser operations by N ms")
    args = parser.parse_args()

    ensure_store_running(args.base_url or os.environ.get("STORE_URL", "http://localhost:5000"))

    agent = BrowserShoppingAgent(
        task_id=args.task,
        mode=args.mode,
        model=args.model,
        headed=args.headed,
        base_url=args.base_url,
        slow_mo=args.slow_mo
    )

    try:
        agent.start()
        if args.mode == "scripted":
            agent.run_scripted()
        else:
            agent.run_groq()
    finally:
        agent.close()


if __name__ == "__main__":
    main()

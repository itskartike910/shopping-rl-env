"""
Run Reporter — Saves a JSON + Markdown report after every agent episode.

Usage (from any agent):
    from reports.run_reporter import save_run_report
    save_run_report(result, agent_type="groq", model="qwen/qwen3.8-27b")

Output files (auto-created in reports/runs/):
    YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.json   — machine-readable full record
    YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.md     — human-readable summary

What each field means in the saved JSON:
    task_id         — integer task number (1-5)
    task_name       — human-readable task description
    agent_type      — "scripted" | "groq" | "ppo"
    model           — LLM model name or "N/A"
    steps           — number of env steps taken in this episode
    total_reward    — cumulative shaped reward from ShoppingEnv
    real_total      — authoritative cart total recalculated from catalog prices
    webshop_score   — normalized score in [0.0, 1.0] following WebShop benchmark
    passed          — bool — did the order satisfy ALL task constraints?
    checks          — dict of individual constraint pass/fail flags from grader
    reason          — human-readable failure reason, or "Task successfully verified"
    order_id        — server-assigned order ID (e.g. "ORD-20261123-001")
    timestamp       — ISO 8601 wall-clock time of this run
"""

import os
import json
import datetime
from typing import Dict, Any, Optional

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _ensure_runs_dir():
    os.makedirs(RUNS_DIR, exist_ok=True)


def save_run_report(
    result: Dict[str, Any],
    agent_type: str = "scripted",
    model: str = "N/A",
    grader_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persist a JSON + Markdown report for one completed episode.

    Args:
        result:        dict returned by agent.run_episode()
                       Expected keys: task_id, steps, total_reward / reward, order
        agent_type:    "scripted" | "groq" | "ppo"
        model:         LLM model name used, or "N/A" for scripted/ppo
        grader_result: Optional pre-computed grader dict. If None, the grader
                       is called here using the order_id from result["order"].

    Returns:
        Path to the saved .json report file.
    """
    _ensure_runs_dir()

    # --- Normalise result keys (scripted uses total_reward, groq uses reward) ---
    task_id = result.get("task_id", 0)
    steps = result.get("steps", 0)
    total_reward = result.get("total_reward", result.get("reward", 0.0))
    order = result.get("order") or {}
    order_id = order.get("order_id", "N/A")

    # --- Run grader if not provided ---
    # Use evaluate_order() directly on the in-memory order dict (works without Flask server).
    # This mirrors how ShoppingEnv.step() calls the grader — no need for orders.json file.
    if grader_result is None and order:
        try:
            from grader.grader import evaluate_order, load_tasks
            tasks = load_tasks()
            task_def = tasks.get(task_id, {})
            constraints = task_def.get("constraints", {})
            grader_result = evaluate_order(order, constraints)
            grader_result["task_name"] = task_def.get("name", f"Task {task_id}")
            grader_result["order_id"] = order_id
        except Exception as e:
            grader_result = {
                "passed": False,
                "webshop_score": 0.0,
                "real_total": order.get("total", 0),
                "checks": {},
                "reason": f"Grader error: {e}",
                "task_name": f"Task {task_id}",
                "order_id": order_id,
            }
    elif grader_result is None:
        grader_result = {
            "passed": False,
            "webshop_score": 0.0,
            "real_total": 0,
            "checks": {},
            "reason": "No order placed — episode ended without checkout",
            "task_name": f"Task {task_id}",
            "order_id": "N/A",
        }

    # --- Compose full record ---
    now = datetime.datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    slug = now.strftime("%Y-%m-%d_%H-%M-%S")

    record = {
        "timestamp": timestamp,
        "task_id": task_id,
        "task_name": grader_result.get("task_name", f"Task {task_id}"),
        "agent_type": agent_type,
        "model": model,
        "steps": steps,
        "total_reward": round(float(total_reward), 4),
        "real_total": grader_result.get("real_total", 0),
        "webshop_score": grader_result.get("webshop_score", 0.0),
        "passed": grader_result.get("passed", False),
        "checks": grader_result.get("checks", {}),
        "reason": grader_result.get("reason", ""),
        "order_id": order_id,
    }

    base_name = f"{slug}_task{task_id}_{agent_type}"
    json_path = os.path.join(RUNS_DIR, f"{base_name}.json")
    md_path = os.path.join(RUNS_DIR, f"{base_name}.md")

    # --- Write JSON ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # --- Write Markdown ---
    passed_badge = "✅ PASSED" if record["passed"] else "❌ FAILED"
    checks_md = "\n".join(
        f"  - {'✅' if v else '❌'} `{k}`" for k, v in record["checks"].items()
    )
    md_content = f"""# Run Report — Task {task_id}: {record['task_name']}

| Field | Value |
|---|---|
| **Status** | {passed_badge} |
| **Timestamp** | {timestamp} |
| **Agent** | {agent_type} |
| **Model** | {model} |
| **Steps** | {steps} |
| **Total Reward** | {record['total_reward']:.4f} |
| **Real Total** | ₹{record['real_total']:,} |
| **WebShop Score** | {record['webshop_score']:.3f} / 1.000 |
| **Order ID** | `{order_id}` |

## Constraint Checks
{checks_md if checks_md else '  *(no checks recorded)*'}

## Reason
> {record['reason']}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[Reporter] Saved report → {json_path}")
    return json_path


def print_summary(record: Dict[str, Any]) -> None:
    """Pretty-print a run record to stdout."""
    status = "✅ PASSED" if record["passed"] else "❌ FAILED"
    print(f"\n{'='*55}")
    print(f"  Task {record['task_id']}: {record['task_name']}")
    print(f"  Agent: {record['agent_type']} | Model: {record['model']}")
    print(f"  Status: {status}")
    print(f"  Steps: {record['steps']} | Reward: {record['total_reward']:.4f}")
    print(f"  Real Total: ₹{record['real_total']:,} | WebShop Score: {record['webshop_score']:.3f}")
    print(f"  Order: {record['order_id']}")
    if not record["passed"]:
        print(f"  Reason: {record['reason']}")
    print(f"{'='*55}\n")

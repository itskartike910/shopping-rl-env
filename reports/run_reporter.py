"""
Run Reporter — Saves a comprehensive JSON + Markdown report after every agent episode.

Logs full run metadata, task instruction, step-by-step trajectory (prompts, thoughts,
observations, actions, rewards, cart state), order summary, and independent grader output.

Output files (auto-created in reports/runs/):
    YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.json   — machine-readable full record
    YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.md     — human-readable detailed report
"""

import os
import json
import datetime
from typing import Dict, Any, Optional, List

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _ensure_runs_dir():
    os.makedirs(RUNS_DIR, exist_ok=True)


def _format_trajectory_md(trajectory: List[Dict[str, Any]]) -> str:
    """Render detailed step-by-step execution trace in Markdown."""
    if not trajectory:
        return "*(no step trajectory recorded)*\n"

    sections = []
    for entry in trajectory:
        step_num = entry.get("step", "?")
        action_str = json.dumps(entry.get("action"), ensure_ascii=False)
        reward = entry.get("reward")
        cum_reward = entry.get("cumulative_reward")
        reward_str = f"{reward:+.4f}" if isinstance(reward, (int, float)) else "N/A"
        cum_str = f"{cum_reward:+.4f}" if isinstance(cum_reward, (int, float)) else "N/A"

        lines = [
            f"### Step {step_num}",
            f"- **Action:** `{action_str}`",
            f"- **Step Reward / Cumulative:** `{reward_str}` / `{cum_str}`",
        ]
        if entry.get("terminated") is not None or entry.get("truncated") is not None:
            lines.append(
                f"- **Terminated / Truncated:** `{entry.get('terminated')}` / `{entry.get('truncated')}`"
            )
        if entry.get("thought"):
            lines.append(f"- **Thought:** {entry['thought']}")
        if entry.get("state") is not None:
            lines.append(
                f"- **State Before Step:** `{json.dumps(entry['state'], ensure_ascii=False)}`"
            )
        if entry.get("obs") is not None:
            lines.append(f"- **Observation:** `{json.dumps(entry['obs'], ensure_ascii=False)}`")
        if entry.get("next_obs") is not None:
            lines.append(f"- **Next Observation:** `{json.dumps(entry['next_obs'], ensure_ascii=False)}`")
        if entry.get("prompt"):
            lines.append("\n**Prompt Sent:**\n```text\n" + str(entry["prompt"]).rstrip() + "\n```")
        if entry.get("raw_response"):
            lines.append("\n**Raw Model Response:**\n```json\n" + str(entry["raw_response"]).rstrip() + "\n```")
        sections.append("\n".join(lines))

    return "\n\n".join(sections) + "\n"


def save_run_report(
    result: Dict[str, Any],
    agent_type: str = "scripted",
    model: str = "N/A",
    grader_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persist a detailed JSON + Markdown report for one completed episode.

    Args:
        result:        dict returned by agent.run_episode()
                       Expected keys: task_id, steps, total_reward / reward, order,
                       plus optional instruction, trajectory, training_summary.
        agent_type:    "scripted" | "groq" | "ppo" | "scripted_browser" | "groq_browser"
        model:         LLM/RL model name used, or "N/A"
        grader_result: Optional pre-computed grader dict. If None, evaluate_order() is invoked.

    Returns:
        Path to the saved .json report file.
    """
    _ensure_runs_dir()

    task_id = result.get("task_id", 0)
    steps = result.get("steps", 0)
    total_reward = result.get("total_reward", result.get("reward", 0.0))
    order = result.get("order") or {}
    order_id = order.get("order_id", "N/A")
    instruction = result.get("instruction", "")
    trajectory = result.get("trajectory", [])
    training_summary = result.get("training_summary")

    # Run independent grader if not supplied
    if grader_result is None and order:
        try:
            from grader.grader import evaluate_order, load_tasks
            tasks = load_tasks()
            task_def = tasks.get(task_id, {})
            if not instruction:
                instruction = task_def.get("instruction", "")
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
        try:
            from grader.grader import load_tasks
            task_def = load_tasks().get(task_id, {})
            if not instruction:
                instruction = task_def.get("instruction", "")
            task_name = task_def.get("name", f"Task {task_id}")
        except Exception:
            task_name = f"Task {task_id}"
        grader_result = {
            "passed": False,
            "webshop_score": 0.0,
            "real_total": 0,
            "checks": {},
            "reason": "No order placed — episode ended without checkout",
            "task_name": task_name,
            "order_id": "N/A",
        }

    now = datetime.datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    slug = now.strftime("%Y-%m-%d_%H-%M-%S")

    record = {
        "timestamp": timestamp,
        "task_id": task_id,
        "task_name": grader_result.get("task_name", f"Task {task_id}"),
        "instruction": instruction,
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
        "order": order if order else None,
        "trajectory": trajectory,
    }
    if training_summary is not None:
        record["training_summary"] = training_summary

    base_name = f"{slug}_task{task_id}_{agent_type}"
    json_path = os.path.join(RUNS_DIR, f"{base_name}.json")
    md_path = os.path.join(RUNS_DIR, f"{base_name}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    passed_badge = "✅ PASSED" if record["passed"] else "❌ FAILED"
    checks_md = "\n".join(
        f"  - {'✅' if v else '❌'} `{k}`" for k, v in record["checks"].items()
    )
    training_md = ""
    if training_summary is not None:
        training_md = f"\n## Training Summary\n```json\n{json.dumps(training_summary, indent=2, ensure_ascii=False)}\n```\n"

    order_md = ""
    if order:
        order_md = f"\n## Order Details\n```json\n{json.dumps(order, indent=2, ensure_ascii=False)}\n```\n"

    trajectory_md = _format_trajectory_md(trajectory)

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

## Task Instruction
> {instruction or 'N/A'}

## Constraint Checks
{checks_md if checks_md else '  *(no checks recorded)*'}

## Reason
> {record['reason']}
{training_md}{order_md}
## Step-by-Step Trajectory
{trajectory_md}"""
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

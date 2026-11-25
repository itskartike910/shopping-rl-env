"""
Authoritative Independent Grader for Shopping RL Environment.

Design Principles:
1. Declarative & Dynamic: Evaluates tasks against schema-defined constraints (tasks.json),
   avoiding fragile hardcoded if/else ladders.
2. Anti-Gaming & Tamper-Resistant: Never trusts agent self-reports or client-side cart totals.
   Always reads server-saved order data and recalculates real price totals from the locked
   authoritative catalog (catalog.json).
3. WebShop Reward Standard: Follows the WebShop benchmark (Yao et al., NeurIPS 2022) attribute
   overlap formulation to compute normalized reward scores in [0.0, 1.0].
"""

import os
import json
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(BASE_DIR, "store", "catalog.json")
ORDERS_PATH = os.path.join(BASE_DIR, "store", "orders.json")
TASKS_PATH = os.path.join(BASE_DIR, "environment", "tasks.json")

def load_locked_catalog() -> Dict[str, Dict[str, Any]]:
    """Load authoritative catalog (single source of truth)."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog_list = json.load(f)
    return {p["id"]: p for p in catalog_list}

def load_tasks() -> Dict[int, Dict[str, Any]]:
    """Load task definitions from tasks.json."""
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks_list = json.load(f)
    return {t["task_id"]: t for t in tasks_list}

def load_order_by_id(order_id: str) -> Dict[str, Any]:
    """Load order record from server-saved orders.json."""
    if not os.path.exists(ORDERS_PATH):
        return None
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            orders = json.load(f)
        return orders.get(order_id)
    except Exception:
        return None

def evaluate_order(order: Dict[str, Any], task_constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically evaluate an order against declarative task constraints.
    Returns:
        {
            "passed": bool,
            "webshop_score": float [0.0 - 1.0],
            "real_total": float,
            "checks": dict of individual constraint outcomes,
            "reason": str
        }
    """
    catalog_by_id = load_locked_catalog()

    # 1. Existence and Status Check
    if not order:
        return {
            "passed": False,
            "webshop_score": 0.0,
            "real_total": 0,
            "checks": {"order_exists": False},
            "reason": "Order does not exist or order_id not found in server database"
        }

    if order.get("status") != "completed" or not order.get("items"):
        return {
            "passed": False,
            "webshop_score": 0.0,
            "real_total": 0,
            "checks": {"order_completed": False},
            "reason": "Order status is not 'completed' or items list is empty"
        }

    # 2. Recalculate Authoritative Total & Validate Stock from Locked Catalog (Anti-Tampering)
    real_total = 0
    validated_items = []
    qty_by_pid = {}
    for item in order["items"]:
        pid = item.get("id")
        qty = item.get("qty", 1)
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {
                "passed": False,
                "webshop_score": 0.0,
                "real_total": real_total,
                "checks": {"valid_quantities": False},
                "reason": f"Invalid quantity {qty} for product ID: {pid}"
            }
        if pid not in catalog_by_id:
            return {
                "passed": False,
                "webshop_score": 0.0,
                "real_total": real_total,
                "checks": {"catalog_integrity": False},
                "reason": f"Order contains unknown product ID: {pid}"
            }
        prod = catalog_by_id[pid]
        qty_by_pid[pid] = qty_by_pid.get(pid, 0) + qty
        real_total += prod["price"] * qty
        validated_items.append({"product": prod, "qty": qty})

    stock_errors = []
    for pid, total_pid_qty in qty_by_pid.items():
        available_stock = int(catalog_by_id[pid].get("stock", 0))
        if total_pid_qty > available_stock:
            stock_errors.append(
                f"Insufficient stock for {pid}: ordered {total_pid_qty}, available {available_stock}"
            )

    checks = {
        "order_exists": True,
        "order_completed": True,
        "valid_quantities": True,
        "catalog_integrity": True,
        "stock_availability": len(stock_errors) == 0
    }
    reasons = list(stock_errors)
    points_earned = 0
    total_points = 0

    # 3. Check Required Product IDs (Exact Match)
    req_pids = task_constraints.get("required_product_ids", [])
    if req_pids:
        total_points += len(req_pids)
        order_pids = {i["product"]["id"] for i in validated_items}
        matched_pids = [pid for pid in req_pids if pid in order_pids]
        checks["required_product_ids"] = len(matched_pids) == len(req_pids)
        points_earned += len(matched_pids)
        if not checks["required_product_ids"]:
            reasons.append(f"Missing required products: {set(req_pids) - order_pids}")

    # 4. Check Required Categories
    req_cats = task_constraints.get("required_categories", [])
    if req_cats:
        total_points += len(req_cats)
        order_cats = {i["product"]["category"] for i in validated_items}
        matched_cats = [c for c in req_cats if c in order_cats]
        checks["required_categories"] = len(matched_cats) == len(req_cats)
        points_earned += len(matched_cats)
        if not checks["required_categories"]:
            reasons.append(f"Missing categories: {set(req_cats) - order_cats}")

    # 5. Check Attribute Requirements
    attr_reqs = task_constraints.get("attribute_requirements", {})
    if attr_reqs:
        for cat, attrs in attr_reqs.items():
            cat_items = [i["product"] for i in validated_items if i["product"]["category"] == cat]
            for attr_name, expected_val in attrs.items():
                total_points += 1
                attr_matched = any(
                    str(item.get(attr_name, "")).lower() == str(expected_val).lower()
                    for item in cat_items
                )
                checks[f"attr_{cat}_{attr_name}"] = attr_matched
                if attr_matched:
                    points_earned += 1
                else:
                    reasons.append(f"Required {cat} with {attr_name}='{expected_val}' not satisfied")

    # 6. Check Socket Compatibility (PC Bundles)
    if task_constraints.get("socket_compatibility", False):
        total_points += 1
        cpus = [i["product"] for i in validated_items if i["product"]["category"] == "CPU"]
        mbs = [i["product"] for i in validated_items if i["product"]["category"] == "Motherboard"]
        compatible = False
        if cpus and mbs:
            cpu_socket = cpus[0].get("socket")
            mb_socket = mbs[0].get("socket")
            compatible = (cpu_socket is not None) and (cpu_socket == mb_socket)
        checks["socket_compatibility"] = compatible
        if compatible:
            points_earned += 1
        else:
            reasons.append("CPU and Motherboard sockets are incompatible or missing")

    # 7. Check Minimum RAM Capacity
    min_ram = task_constraints.get("min_ram_gb")
    if min_ram:
        total_points += 1
        total_ram = sum(
            i["product"].get("ram_gb", 0) * i["qty"]
            for i in validated_items
            if i["product"]["category"] == "RAM"
        )
        checks["min_ram_gb"] = total_ram >= min_ram
        if checks["min_ram_gb"]:
            points_earned += 1
        else:
            reasons.append(f"RAM capacity {total_ram}GB below required {min_ram}GB")

    # 8. Check Minimum Storage Capacity
    min_storage = task_constraints.get("min_storage_gb")
    if min_storage:
        total_points += 1
        total_storage = sum(
            i["product"].get("storage_gb", 0) * i["qty"]
            for i in validated_items
            if i["product"]["category"] == "SSD"
        )
        checks["min_storage_gb"] = total_storage >= min_storage
        if checks["min_storage_gb"]:
            points_earned += 1
        else:
            reasons.append(f"Storage capacity {total_storage}GB below required {min_storage}GB")

    # 9. Check Item Counts
    min_items = task_constraints.get("min_item_count", 1)
    max_items = task_constraints.get("max_item_count", 99)
    total_qty = sum(i["qty"] for i in validated_items)
    checks["item_count_range"] = (min_items <= total_qty <= max_items)
    if not checks["item_count_range"]:
        reasons.append(f"Item count {total_qty} outside allowed range [{min_items}, {max_items}]")

    # 10. Check Budget (Hard constraint in WebShop)
    max_budget = task_constraints.get("max_budget")
    budget_ok = True
    if max_budget is not None:
        budget_ok = real_total <= max_budget
        checks["budget_constraint"] = budget_ok
        if not budget_ok:
            reasons.append(f"Total price ₹{real_total:,} exceeds budget ₹{max_budget:,}")

    # WebShop-style normalized score computation:
    # Partial score from satisfied attributes * 1.0 if budget_ok else 0.0
    if total_points > 0:
        attribute_overlap = points_earned / total_points
    else:
        attribute_overlap = 1.0

    webshop_score = round(attribute_overlap if budget_ok else (attribute_overlap * 0.2), 3)

    # Full pass requires all checks to be True
    passed = all(checks.values()) and budget_ok

    return {
        "passed": passed,
        "webshop_score": 1.0 if passed else webshop_score,
        "real_total": real_total,
        "checks": checks,
        "reason": "Task successfully verified" if passed else "; ".join(reasons)
    }

def grade_task(task_id: int, order_id: str) -> Dict[str, Any]:
    """Grade a specific task by task ID and order ID."""
    tasks = load_tasks()
    if task_id not in tasks:
        raise ValueError(f"Unknown task_id: {task_id}")
    task = tasks[task_id]
    order = load_order_by_id(order_id)
    result = evaluate_order(order, task["constraints"])
    result["task_id"] = task_id
    result["task_name"] = task["name"]
    result["order_id"] = order_id
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m grader.grader <task_id> <order_id>")
        sys.exit(1)
    t_id = int(sys.argv[1])
    o_id = sys.argv[2]
    res = grade_task(t_id, o_id)
    print(json.dumps(res, indent=2))

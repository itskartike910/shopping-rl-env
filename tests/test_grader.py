"""
Unit tests for the Declarative Independent Grader across all 5 tasks.
Verifies that valid orders pass with score 1.0 and invalid orders fail.
"""

import pytest
from grader.grader import evaluate_order, load_tasks

@pytest.fixture
def tasks():
    return load_tasks()

def test_task_1_direct_purchase(tasks):
    # Task 1: Buy 1x BOT-01 (Milton Bottle)
    task1 = tasks[1]
    valid_order = {
        "status": "completed",
        "items": [{"id": "BOT-01", "qty": 1}]
    }
    result = evaluate_order(valid_order, task1["constraints"])
    assert result["passed"] is True
    assert result["webshop_score"] == 1.0
    assert result["real_total"] == 599

    # Wrong product
    invalid_order = {
        "status": "completed",
        "items": [{"id": "BOT-02", "qty": 1}]
    }
    res_invalid = evaluate_order(invalid_order, task1["constraints"])
    assert res_invalid["passed"] is False

def test_task_2_constrained_wireless_mouse(tasks):
    # Task 2: Black wireless mouse under ₹5,000
    task2 = tasks[2]
    # Valid: MOU-01 is Logitech G305 (Wireless, Black, ₹2,999)
    valid_order = {
        "status": "completed",
        "items": [{"id": "MOU-01", "qty": 1}]
    }
    result = evaluate_order(valid_order, task2["constraints"])
    assert result["passed"] is True
    assert result["real_total"] == 2999

    # Invalid: MOU-03 is Logitech MX Master 3S (Wireless, Graphite, ₹8,999) -> Over budget
    over_budget_order = {
        "status": "completed",
        "items": [{"id": "MOU-03", "qty": 1}]
    }
    res_ob = evaluate_order(over_budget_order, task2["constraints"])
    assert res_ob["passed"] is False

def test_task_3_keyboard_mouse_bundle(tasks):
    # Task 3: Wired mechanical keyboard + wired mouse under ₹5,000
    task3 = tasks[3]
    # KEY-02 (Redragon Mechanical Wired, ₹2,799) + MOU-04 (Cosmic Byte Wired, ₹799) = ₹3,598
    valid_order = {
        "status": "completed",
        "items": [{"id": "KEY-02", "qty": 1}, {"id": "MOU-04", "qty": 1}]
    }
    result = evaluate_order(valid_order, task3["constraints"])
    assert result["passed"] is True
    assert result["real_total"] == 3598

    # Over-budget: KEY-02 (₹2,799) + MOU-02 (Razer Wired, ₹4,499) = ₹7,298
    over_budget_order = {
        "status": "completed",
        "items": [{"id": "KEY-02", "qty": 1}, {"id": "MOU-02", "qty": 1}]
    }
    res_ob = evaluate_order(over_budget_order, task3["constraints"])
    assert res_ob["passed"] is False

def test_task_4_socket_compatibility(tasks):
    # Task 4: Compatible CPU + Motherboard + min 16GB RAM under ₹45,000
    task4 = tasks[4]
    # CPU-01 (Intel i5-13400, LGA1700, ₹18,999) + MB-01 (B660M LGA1700, ₹14,999) + RAM-02 (16GB, ₹4,799)
    # Total = ₹38,797
    compatible_order = {
        "status": "completed",
        "items": [{"id": "CPU-01", "qty": 1}, {"id": "MB-01", "qty": 1}, {"id": "RAM-02", "qty": 1}]
    }
    res_comp = evaluate_order(compatible_order, task4["constraints"])
    assert res_comp["passed"] is True
    assert res_comp["checks"]["socket_compatibility"] is True

    # Incompatible: CPU-02 (AMD AM5) + MB-01 (Intel LGA1700)
    incompatible_order = {
        "status": "completed",
        "items": [{"id": "CPU-02", "qty": 1}, {"id": "MB-01", "qty": 1}, {"id": "RAM-02", "qty": 1}]
    }
    res_incomp = evaluate_order(incompatible_order, task4["constraints"])
    assert res_incomp["passed"] is False
    assert res_incomp["checks"]["socket_compatibility"] is False

def test_task_5_full_gaming_pc(tasks):
    # Task 5: 10 PC categories under ₹110,000
    task5 = tasks[5]
    valid_pc_order = {
        "status": "completed",
        "items": [
            {"id": "CPU-01", "qty": 1}, # 18,999 (LGA1700)
            {"id": "MB-01", "qty": 1},  # 14,999 (LGA1700)
            {"id": "RAM-02", "qty": 1}, #  4,799 (16GB)
            {"id": "SSD-01", "qty": 1}, #  4,999 (500GB)
            {"id": "GPU-02", "qty": 1}, # 29,999 (RX 7600)
            {"id": "CAB-01", "qty": 1}, #  8,499
            {"id": "PSU-01", "qty": 1}, #  6,999
            {"id": "MON-01", "qty": 1}, # 14,999
            {"id": "KEY-02", "qty": 1}, #  2,799
            {"id": "MOU-01", "qty": 1}  #  2,999
            # Total = ₹109,090 (< ₹110,000)
        ]
    }
    res = evaluate_order(valid_pc_order, task5["constraints"])
    assert res["passed"] is True
    assert res["real_total"] == 110090

"""
Unit tests for Flask Store inventory and checkout stock enforcement.
"""

import pytest
import store.app as store_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide an isolated Flask test client with reset stock and temp orders file."""
    orders_file = tmp_path / "orders.json"
    orders_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(store_app, "ORDERS_PATH", str(orders_file))
    store_app.SESSION_CARTS.clear()
    store_app.reset_catalog_stock()
    with store_app.app.test_client() as test_client:
        yield test_client
    store_app.SESSION_CARTS.clear()
    store_app.reset_catalog_stock()


def test_api_add_to_cart_rejects_above_stock(client):
    """MB-01 has stock=5; adding 6 at once or cumulatively must be rejected."""
    headers = {"X-Session-ID": "sess-stock-1"}

    res_too_many = client.post(
        "/api/cart/add",
        json={"product_id": "MB-01", "qty": 6},
        headers=headers,
    )
    assert res_too_many.status_code == 400
    assert "exceeds available stock (5)" in res_too_many.get_json()["message"]

    res_ok = client.post(
        "/api/cart/add",
        json={"product_id": "MB-01", "qty": 3},
        headers=headers,
    )
    assert res_ok.status_code == 200

    res_cumulative_overflow = client.post(
        "/api/cart/add",
        json={"product_id": "MB-01", "qty": 3},
        headers=headers,
    )
    assert res_cumulative_overflow.status_code == 400
    assert "exceeds available stock (5)" in res_cumulative_overflow.get_json()["message"]

    cart = client.get("/api/cart", headers=headers).get_json()
    assert cart["item_count"] == 3


def test_api_checkout_checks_stock_and_decrements(client):
    """
    Two sessions each reserve 4 units of MB-01 (stock=5).
    First checkout succeeds (remaining stock=1); second checkout must be blocked at checkout time.
    """
    sess_a = {"X-Session-ID": "sess-a"}
    sess_b = {"X-Session-ID": "sess-b"}

    assert client.post("/api/cart/add", json={"product_id": "MB-01", "qty": 4}, headers=sess_a).status_code == 200
    assert client.post("/api/cart/add", json={"product_id": "MB-01", "qty": 4}, headers=sess_b).status_code == 200

    checkout_a = client.post("/api/checkout", headers=sess_a)
    assert checkout_a.status_code == 200
    assert checkout_a.get_json()["status"] == "completed"
    assert store_app.CATALOG_BY_ID["MB-01"]["stock"] == 1

    checkout_b = client.post("/api/checkout", headers=sess_b)
    assert checkout_b.status_code == 400
    body_b = checkout_b.get_json()
    assert body_b["status"] == "failed"
    assert body_b["order_id"] is None
    assert "Insufficient stock" in body_b["message"]
    assert "available 1" in body_b["message"]
    # Stock untouched and Session B cart kept intact for adjustment
    assert store_app.CATALOG_BY_ID["MB-01"]["stock"] == 1
    assert client.get("/api/cart", headers=sess_b).get_json()["item_count"] == 4


def test_ui_checkout_blocks_overstock_and_marks_out_of_stock(client):
    """HTML UI checkout blocks insufficient stock and renders Out of Stock when depleted."""
    # Buy all 5 units of MB-01 via UI
    client.post("/cart/add", data={"product_id": "MB-01", "qty": 5}, follow_redirects=True)
    ok_checkout = client.post("/checkout", follow_redirects=True)
    assert ok_checkout.status_code == 200
    assert b"Order Placed Successfully!" in ok_checkout.data
    assert store_app.CATALOG_BY_ID["MB-01"]["stock"] == 0

    # Index page should show Out of Stock for MB-01
    index_res = client.get("/?q=MB-01")
    assert b"Out of Stock" in index_res.data

    # Attempting to add when out of stock fails cleanly with flash error
    add_oob = client.post("/cart/add", data={"product_id": "MB-01", "qty": 1}, follow_redirects=True)
    assert b"out of stock" in add_oob.data

    # Injecting stale/over-stock cart item directly and hitting UI /checkout must block with 400
    with client.session_transaction() as flask_sess:
        s_id = flask_sess["session_id"]
    store_app.SESSION_CARTS[s_id] = [{"id": "MB-01", "qty": 2}]

    blocked = client.post("/checkout")
    assert blocked.status_code == 400
    assert b"Insufficient stock" in blocked.data


def test_browser_finish_run_fetches_order_from_store_api(monkeypatch, tmp_path):
    """BrowserShoppingAgent._finish_run should resolve orders via /api/orders/<id> even if local orders.json is empty."""
    import io
    import json
    import agents.browser_agent as browser_mod
    import grader.grader as grader_mod
    import reports.run_reporter as reporter_mod

    # Ensure local orders.json lookup has nothing
    empty_orders = tmp_path / "orders.json"
    empty_orders.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(grader_mod, "ORDERS_PATH", str(empty_orders))
    monkeypatch.setattr(reporter_mod, "RUNS_DIR", str(tmp_path / "runs"))

    api_payload = {
        "status": "success",
        "order": {
            "order_id": "ORD-REMOTE1",
            "status": "completed",
            "items": [{"id": "BOT-01", "price": 599, "qty": 1}],
            "total": 599,
        },
    }

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import urllib.request
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=2.0: _FakeResp(json.dumps(api_payload).encode("utf-8")),
    )

    ag = browser_mod.BrowserShoppingAgent(task_id=1, mode="scripted", base_url="http://127.0.0.1:5000")
    res = ag._finish_run("ORD-REMOTE1", steps=3, agent_type="scripted_browser")
    assert res["total_reward"] == 1.5
    assert res["order"]["order_id"] == "ORD-REMOTE1"

"""
Flask Mini E-Commerce Store for Shopping RL Environment.
Serves both a REST API for the Gym environment and HTML templates for Playwright / human inspection.
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "catalog.json")
ORDERS_PATH = os.path.join(BASE_DIR, "orders.json")

app = Flask(__name__)
app.secret_key = "shopping-rl-secret-key-eval-env"

# Load catalog once (read-only reference on disk; runtime stock tracked in memory)
with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)
CATALOG_BY_ID = {p["id"]: p for p in CATALOG}
INITIAL_STOCK = {p["id"]: int(p.get("stock", 0)) for p in CATALOG}

# In-memory session carts: session_id -> list of {"id": product_id, "qty": int}
SESSION_CARTS = {}


def reset_catalog_stock():
    """Restore in-memory product stock levels back to catalog.json defaults."""
    for pid, stock_val in INITIAL_STOCK.items():
        if pid in CATALOG_BY_ID:
            CATALOG_BY_ID[pid]["stock"] = stock_val

def get_session_id():
    """Extract session ID from request headers, query args, or flask session."""
    s_id = request.headers.get("X-Session-ID") or request.args.get("session_id")
    if not s_id:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())[:8]
        s_id = session["session_id"]
    if s_id not in SESSION_CARTS:
        SESSION_CARTS[s_id] = []
    return s_id

def load_orders():
    """Load orders from orders.json safely."""
    if not os.path.exists(ORDERS_PATH):
        return {}
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_order(order_data):
    """Save an order directly to server-side orders.json."""
    orders = load_orders()
    orders[order_data["order_id"]] = order_data
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2)

def filter_and_sort_products(query="", category="", sort_by="default", min_price=None, max_price=None):
    """Filter and sort catalog products."""
    results = CATALOG.copy()
    if query and query.lower() != "all":
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
    if category and category.lower() != "all":
        results = [p for p in results if p["category"].lower() == category.lower()]
    if min_price is not None:
        results = [p for p in results if p["price"] >= min_price]
    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]

    if sort_by == "price_asc":
        results.sort(key=lambda x: x["price"])
    elif sort_by == "price_desc":
        results.sort(key=lambda x: x["price"], reverse=True)
    elif sort_by == "rating":
        results.sort(key=lambda x: x.get("rating", 0), reverse=True)
    elif sort_by == "name":
        results.sort(key=lambda x: x["name"])

    return results

def _build_cart_details(s_id, include_full_product=False):
    """Build itemized cart lines and authoritative total from CATALOG_BY_ID."""
    cart_items = SESSION_CARTS.get(s_id, [])
    detailed = []
    total = 0
    for item in cart_items:
        prod = CATALOG_BY_ID.get(item["id"])
        if not prod:
            continue
        subtotal = prod["price"] * item["qty"]
        total += subtotal
        if include_full_product:
            detailed.append({**prod, "qty": item["qty"], "subtotal": subtotal})
        else:
            detailed.append({
                "id": prod["id"],
                "name": prod["name"],
                "category": prod["category"],
                "price": prod["price"],
                "qty": item["qty"],
                "subtotal": subtotal,
                "available_stock": prod.get("stock", 0),
            })
    item_count = sum(i["qty"] for i in cart_items)
    return detailed, total, item_count


def _add_item_to_cart(s_id, product_id, raw_qty=1):
    """Validate product, positive qty, and available stock before updating session cart."""
    pid = str(product_id or "").upper().strip()
    try:
        qty = int(raw_qty)
    except (TypeError, ValueError):
        return False, "Quantity must be a valid integer", 400

    if pid not in CATALOG_BY_ID:
        return False, f"Invalid product_id: {pid}", 400
    if qty <= 0:
        return False, "Quantity must be positive", 400

    prod = CATALOG_BY_ID[pid]
    available_stock = int(prod.get("stock", 0))
    cart = SESSION_CARTS.setdefault(s_id, [])
    existing = next((item for item in cart if item["id"] == pid), None)
    existing_qty = existing["qty"] if existing else 0
    new_qty = existing_qty + qty

    if available_stock <= 0:
        return False, f"{prod['name']} ({pid}) is out of stock", 400
    if new_qty > available_stock:
        return (
            False,
            f"Cannot add {qty} more of {prod['name']} ({pid}): requested total {new_qty} exceeds available stock ({available_stock}).",
            400,
        )

    if existing:
        existing["qty"] = new_qty
    else:
        cart.append({"id": pid, "qty": qty})
    return True, f"Added {pid} to cart", 200


def _process_checkout(s_id):
    """Validate cart & live stock at checkout time, decrement stock, and persist order."""
    cart_items = SESSION_CARTS.get(s_id, [])
    if not cart_items:
        return None, "Cannot checkout with empty cart", 400

    stock_errors = []
    order_items = []
    total = 0

    for item in cart_items:
        pid = item.get("id")
        qty = int(item.get("qty", 0))
        prod = CATALOG_BY_ID.get(pid)
        if not prod:
            stock_errors.append(f"Unknown product {pid} in cart.")
            continue
        available_stock = int(prod.get("stock", 0))
        if qty <= 0:
            stock_errors.append(f"Invalid quantity ({qty}) for {prod['name']} ({pid}).")
            continue
        if qty > available_stock:
            stock_errors.append(
                f"Insufficient stock for {prod['name']} ({pid}): ordered {qty}, available {available_stock}."
            )
            continue

        subtotal = prod["price"] * qty
        total += subtotal
        order_items.append({
            "id": prod["id"],
            "name": prod["name"],
            "category": prod["category"],
            "price": prod["price"],
            "qty": qty,
            "subtotal": subtotal,
        })

    if stock_errors:
        return None, " ".join(stock_errors), 400

    # Deduct stock only after every cart item passes validation
    for item in order_items:
        CATALOG_BY_ID[item["id"]]["stock"] -= item["qty"]

    order_id = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    order_data = {
        "order_id": order_id,
        "session_id": s_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "completed",
        "items": order_items,
        "total": total,
        "item_count": sum(i["qty"] for i in order_items),
    }

    save_order(order_data)
    SESSION_CARTS[s_id] = []
    return order_data, "Order placed successfully", 200


# ----------------- REST API ENDPOINTS -----------------

@app.route("/api/catalog", methods=["GET"])
def api_catalog():
    return jsonify({"status": "success", "total": len(CATALOG), "products": CATALOG})

@app.route("/api/search", methods=["GET"])
def api_search():
    q = request.args.get("q", "")
    category = request.args.get("category", "")
    sort_by = request.args.get("sort_by", "default")
    min_p = request.args.get("min_price", type=float)
    max_p = request.args.get("max_price", type=float)

    results = filter_and_sort_products(q, category, sort_by, min_p, max_p)
    return jsonify({
        "status": "success",
        "query": q,
        "category": category,
        "sort_by": sort_by,
        "count": len(results),
        "products": results
    })

@app.route("/api/product/<product_id>", methods=["GET"])
def api_product(product_id):
    pid = product_id.upper().strip()
    if pid not in CATALOG_BY_ID:
        return jsonify({"status": "error", "message": f"Product {pid} not found"}), 404
    return jsonify({"status": "success", "product": CATALOG_BY_ID[pid]})

@app.route("/api/cart", methods=["GET"])
def api_get_cart():
    s_id = get_session_id()
    detailed_items, total, item_count = _build_cart_details(s_id, include_full_product=False)
    return jsonify({
        "status": "success",
        "session_id": s_id,
        "items": detailed_items,
        "item_count": item_count,
        "total": total
    })

@app.route("/api/cart/add", methods=["POST"])
def api_add_to_cart():
    s_id = get_session_id()
    data = request.get_json(force=True, silent=True) or request.form
    ok, msg, status_code = _add_item_to_cart(s_id, data.get("product_id", ""), data.get("qty", 1))
    if not ok:
        return jsonify({"status": "error", "message": msg}), status_code
    return jsonify({"status": "success", "message": msg, "cart_size": len(SESSION_CARTS.get(s_id, []))})

@app.route("/api/cart/remove", methods=["POST"])
def api_remove_from_cart():
    s_id = get_session_id()
    data = request.get_json(force=True, silent=True) or request.form
    product_id = data.get("product_id", "").upper().strip()

    cart = SESSION_CARTS.setdefault(s_id, [])
    initial_len = len(cart)
    SESSION_CARTS[s_id] = [item for item in cart if item["id"] != product_id]

    removed = initial_len > len(SESSION_CARTS[s_id])
    return jsonify({"status": "success" if removed else "not_found", "product_id": product_id})

@app.route("/api/cart/clear", methods=["POST"])
def api_clear_cart():
    s_id = get_session_id()
    SESSION_CARTS[s_id] = []
    return jsonify({"status": "success", "message": "Cart cleared"})

@app.route("/api/checkout", methods=["POST"])
def api_checkout():
    s_id = get_session_id()
    order_data, message, status_code = _process_checkout(s_id)
    if order_data is None:
        return jsonify({
            "status": "failed",
            "message": message,
            "order_id": None
        }), status_code

    return jsonify({
        "status": "completed",
        "message": message,
        "order": order_data
    })

@app.route("/api/reset", methods=["POST"])
def api_reset():
    s_id = get_session_id()
    SESSION_CARTS[s_id] = []
    data = request.get_json(force=True, silent=True) or {}
    if data.get("reset_stock"):
        reset_catalog_stock()
    return jsonify({"status": "success", "session_id": s_id, "message": "Session reset"})

@app.route("/api/orders/<order_id>", methods=["GET"])
def api_get_order(order_id):
    orders = load_orders()
    if order_id not in orders:
        return jsonify({"status": "error", "message": "Order not found"}), 404
    return jsonify({"status": "success", "order": orders[order_id]})

# ----------------- HTML WEB UI (PLAYWRIGHT / HUMAN) -----------------

@app.route("/", methods=["GET"])
def ui_index():
    s_id = get_session_id()
    q = request.args.get("q", "")
    category = request.args.get("category", "")
    sort_by = request.args.get("sort_by", "default")

    products = filter_and_sort_products(q, category, sort_by)
    categories = sorted(list(set(p["category"] for p in CATALOG)))
    cart_count = sum(i["qty"] for i in SESSION_CARTS.get(s_id, []))

    return render_template(
        "index.html",
        products=products,
        categories=categories,
        current_category=category,
        query=q,
        sort_by=sort_by,
        cart_count=cart_count,
        session_id=s_id
    )

@app.route("/product/<product_id>", methods=["GET"])
def ui_product(product_id):
    s_id = get_session_id()
    pid = product_id.upper().strip()
    prod = CATALOG_BY_ID.get(pid)
    if not prod:
        return "Product not found", 404
    cart_count = sum(i["qty"] for i in SESSION_CARTS.get(s_id, []))
    return render_template("product.html", product=prod, cart_count=cart_count, session_id=s_id)

@app.route("/cart", methods=["GET"])
def ui_cart():
    s_id = get_session_id()
    detailed, total, _ = _build_cart_details(s_id, include_full_product=True)
    return render_template("cart.html", items=detailed, total=total, session_id=s_id)

@app.route("/cart/add", methods=["POST"])
def ui_cart_add():
    s_id = get_session_id()
    ok, msg, _ = _add_item_to_cart(s_id, request.form.get("product_id", ""), request.form.get("qty", 1))
    if not ok:
        flash(msg, "error")
    return redirect(request.referrer or url_for("ui_cart"))

@app.route("/cart/remove", methods=["POST"])
def ui_cart_remove():
    s_id = get_session_id()
    pid = request.form.get("product_id", "").upper().strip()
    SESSION_CARTS[s_id] = [i for i in SESSION_CARTS.get(s_id, []) if i["id"] != pid]
    return redirect(url_for("ui_cart"))

@app.route("/checkout", methods=["POST"])
def ui_checkout():
    s_id = get_session_id()
    order_data, message, status_code = _process_checkout(s_id)
    if order_data is None:
        if not SESSION_CARTS.get(s_id):
            return redirect(url_for("ui_cart"))
        detailed, total, _ = _build_cart_details(s_id, include_full_product=True)
        return render_template("cart.html", items=detailed, total=total, session_id=s_id, error=message), status_code
    return redirect(url_for("ui_checkout_success", order_id=order_data["order_id"]))

@app.route("/checkout-success/<order_id>", methods=["GET"])
def ui_checkout_success(order_id):
    orders = load_orders()
    order = orders.get(order_id)
    if not order:
        return "Order not found", 404
    return render_template("checkout.html", order=order)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

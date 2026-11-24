"""
Flask Mini E-Commerce Store for Shopping RL Environment.
Serves both a REST API for the Gym environment and HTML templates for Playwright / human inspection.
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "catalog.json")
ORDERS_PATH = os.path.join(BASE_DIR, "orders.json")

app = Flask(__name__)
app.secret_key = "shopping-rl-secret-key-eval-env"

# Load catalog once (read-only reference)
with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)
CATALOG_BY_ID = {p["id"]: p for p in CATALOG}

# In-memory session carts: session_id -> list of {"id": product_id, "qty": int}
SESSION_CARTS = {}

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
    cart_items = SESSION_CARTS.get(s_id, [])
    detailed_items = []
    total = 0
    for item in cart_items:
        prod = CATALOG_BY_ID.get(item["id"])
        if prod:
            subtotal = prod["price"] * item["qty"]
            total += subtotal
            detailed_items.append({
                "id": prod["id"],
                "name": prod["name"],
                "category": prod["category"],
                "price": prod["price"],
                "qty": item["qty"],
                "subtotal": subtotal
            })
    return jsonify({
        "status": "success",
        "session_id": s_id,
        "items": detailed_items,
        "item_count": sum(i["qty"] for i in cart_items),
        "total": total
    })

@app.route("/api/cart/add", methods=["POST"])
def api_add_to_cart():
    s_id = get_session_id()
    data = request.get_json(force=True, silent=True) or request.form
    product_id = data.get("product_id", "").upper().strip()
    qty = int(data.get("qty", 1))

    if product_id not in CATALOG_BY_ID:
        return jsonify({"status": "error", "message": f"Invalid product_id: {product_id}"}), 400
    if qty <= 0:
        return jsonify({"status": "error", "message": "Quantity must be positive"}), 400

    cart = SESSION_CARTS.setdefault(s_id, [])
    # Check if already in cart
    existing = next((item for item in cart if item["id"] == product_id), None)
    if existing:
        existing["qty"] += qty
    else:
        cart.append({"id": product_id, "qty": qty})

    return jsonify({"status": "success", "message": f"Added {product_id} to cart", "cart_size": len(cart)})

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
    cart_items = SESSION_CARTS.get(s_id, [])

    if not cart_items:
        return jsonify({
            "status": "failed",
            "message": "Cannot checkout with empty cart",
            "order_id": None
        }), 400

    # Calculate real authoritative total from locked CATALOG
    order_items = []
    total = 0
    for item in cart_items:
        prod = CATALOG_BY_ID.get(item["id"])
        if prod:
            subtotal = prod["price"] * item["qty"]
            total += subtotal
            order_items.append({
                "id": prod["id"],
                "name": prod["name"],
                "category": prod["category"],
                "price": prod["price"],
                "qty": item["qty"],
                "subtotal": subtotal
            })

    order_id = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    order_data = {
        "order_id": order_id,
        "session_id": s_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "completed",
        "items": order_items,
        "total": total,
        "item_count": sum(i["qty"] for i in order_items)
    }

    save_order(order_data)
    # Clear cart after successful checkout
    SESSION_CARTS[s_id] = []

    return jsonify({
        "status": "completed",
        "message": "Order placed successfully",
        "order": order_data
    })

@app.route("/api/reset", methods=["POST"])
def api_reset():
    s_id = get_session_id()
    SESSION_CARTS[s_id] = []
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
    cart_items = SESSION_CARTS.get(s_id, [])
    detailed = []
    total = 0
    for item in cart_items:
        prod = CATALOG_BY_ID.get(item["id"])
        if prod:
            subtotal = prod["price"] * item["qty"]
            total += subtotal
            detailed.append({**prod, "qty": item["qty"], "subtotal": subtotal})
    return render_template("cart.html", items=detailed, total=total, session_id=s_id)

@app.route("/cart/add", methods=["POST"])
def ui_cart_add():
    s_id = get_session_id()
    pid = request.form.get("product_id", "").upper().strip()
    qty = int(request.form.get("qty", 1))
    if pid in CATALOG_BY_ID and qty > 0:
        cart = SESSION_CARTS.setdefault(s_id, [])
        item = next((i for i in cart if i["id"] == pid), None)
        if item:
            item["qty"] += qty
        else:
            cart.append({"id": pid, "qty": qty})
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
    cart_items = SESSION_CARTS.get(s_id, [])
    if not cart_items:
        return redirect(url_for("ui_cart"))

    order_items = []
    total = 0
    for item in cart_items:
        prod = CATALOG_BY_ID.get(item["id"])
        if prod:
            subtotal = prod["price"] * item["qty"]
            total += subtotal
            order_items.append({
                "id": prod["id"],
                "name": prod["name"],
                "category": prod["category"],
                "price": prod["price"],
                "qty": item["qty"],
                "subtotal": subtotal
            })

    order_id = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    order_data = {
        "order_id": order_id,
        "session_id": s_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "completed",
        "items": order_items,
        "total": total,
        "item_count": sum(i["qty"] for i in order_items)
    }
    save_order(order_data)
    SESSION_CARTS[s_id] = []
    return redirect(url_for("ui_checkout_success", order_id=order_id))

@app.route("/checkout-success/<order_id>", methods=["GET"])
def ui_checkout_success(order_id):
    orders = load_orders()
    order = orders.get(order_id)
    if not order:
        return "Order not found", 404
    return render_template("checkout.html", order=order)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

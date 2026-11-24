# 🛒 Shopping RL Environment (`shopping-rl-env`)

> **Lead Reinforcement Learning Environment Engineer Take-Home Assignment**  
> **Adzzat Labs, Mumbai**  
> Built for evaluating, training, and benchmarking autonomous shopping agents under compositional constraints.

---

## ⚡ Quickstart: Run in One Command

### 1. Setup Virtual Environment (`.venv`)
```bash
cd shopping-rl-env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. One Command to Run Baseline Agent (All 5 Tasks)
```bash
PYTHONPATH=. python agents/scripted_agent.py
```

### 3. Run Headed Chromium Browser Agent (Watch it on Screen!)
You can watch the browser visually click, search, add products, and checkout in real time:

```bash
# 1. Run Scripted Agent in Headed Chromium (Task 1)
PYTHONPATH=. python agents/browser_agent.py --task 1 --mode scripted --headed

# 2. Run Scripted Agent on Task 2 (Wireless Mouse)
PYTHONPATH=. python agents/browser_agent.py --task 2 --mode scripted --headed

# 3. Run Groq LLM Agent in Headed Chromium (AI reads screen & drives browser)
PYTHONPATH=. python agents/browser_agent.py --task 1 --mode groq --headed

# (Omit --headed to run in headless background mode)
```

### 4. Run Headed Chromium Using Docker
To let Docker show a visible browser window on your host display:

```bash
# Step 1: Allow Docker to access your X11 display (run on your Linux terminal)
xhost +local:docker

# Step 2: Run Scripted Agent headed inside Docker
docker compose run --rm agent-eval python agents/browser_agent.py --task 1 --mode scripted --headed

# Step 3: Run Groq LLM Agent headed inside Docker
docker compose run --rm agent-eval python agents/browser_agent.py --task 1 --mode groq --headed
```

### 5. Run Fast Headless RL Training & Evaluation
```bash
# Groq LLM in Gym environment
PYTHONPATH=. python agents/groq_agent.py 1 qwen/qwen3.8-27b

# Policy Gradient RL trainer
PYTHONPATH=. python agents/ppo_agent.py 1
```

### 6. One Command to Run Full Test Suite (12 Tests)
```bash
PYTHONPATH=. pytest tests/ -v
```

### 7. Run Standard Docker Compose
```bash
# Starts the store and runs the baseline agent
docker compose up --build
```

---

## 📖 What Each Code File & Function Does

This project is built from scratch with clean, modular architecture. Here is what every component does:

### 1. Store Backend & Catalog
- **`store/catalog.json`**: The authoritative **single source of truth** containing 32 technical products (CPUs, motherboards, RAM, SSDs, GPUs, PC cases, PSUs, monitors, keyboards, mice, audio equipment, bottles, and backpacks). Contains rich metadata: `id`, `name`, `price`, `stock`, `rating`, `color`, `connectivity`, `type`, `socket` (e.g. `LGA1700`, `AM5`), `ram_gb`, `storage_gb`, and search `tags`.
- **`store/app.py`**: Lightweight Flask e-commerce application serving both:
  - **REST API** (`/api/search`, `/api/catalog`, `/api/product/<id>`, `/api/cart`, `/api/checkout`, `/api/reset`): for fast in-process interactions.
  - **HTML Web Pages** (`/`, `/product/<id>`, `/cart`, `/checkout`): responsive UI for human inspection and Playwright browser agents.
  - **Key functions:**
    - `filter_and_sort_products()`: filters by search term, category, price bounds, and sorts by price asc/desc, rating, or name.
    - `api_checkout()`: calculates real totals, assigns an order ID, and commits to `orders.json`.
- **`store/templates/`**:
  - `layout.html`: base HTML template with navigation bar, responsive CSS styling, and live cart item counter.
  - `index.html`: store front page with search bar, sort dropdown, category filter pills, and product cards with "Add to Cart" buttons.
  - `product.html`: detailed product specification view showing socket compatibility, switch type, wattage, and stock status.
  - `cart.html`: shopping cart table displaying item breakdown, subtotal, and checkout trigger.
  - `checkout.html`: confirmation page displaying authoritative confirmed order ID and line items.

### 2. Environment Layer
- **`environment/tasks.json`**: Declarative schema for all 5 tasks. Defines difficulty, natural language prompt, step limits, required product IDs, required categories, attribute constraints, socket compatibility requirements, and budget ceilings.
- **`environment/env.py` (`ShoppingEnv`)**: Farama Gymnasium-compliant RL environment (`gym.Env`).
  - **Action Space:** `MultiDiscrete([8, 37])` supporting 8 action types:
    - `0: SEARCH`: query by keyword.
    - `1: SORT_BY`: sort by price low-to-high, high-to-low, rating, name.
    - `2: FILTER_CATEGORY`: filter by category.
    - `3: VIEW_PRODUCT`: inspect product specifications.
    - `4: ADD_TO_CART`: add product to session cart.
    - `5: REMOVE_FROM_CART`: remove item from cart.
    - `6: CLEAR_CART`: reset cart.
    - `7: CHECKOUT`: finalize purchase and terminate episode.
  - **Observation Space:** 21-dimensional normalized `Box` space (step progress, cart size, spend ratio, current page, 16 category presence flags, action validity flag).
  - **`_get_info()`**: provides structured natural language state for LLMs (current cart, items, prices, visible products).
  - **`step()`**: executes action, enforces step limits, applies WebShop trajectory penalties, and calls the authoritative grader upon checkout.
- **`environment/browser_env.py` (`BrowserShoppingAgent`)**: Optional Playwright Chromium adapter that launches headless or headed browser windows, clicks buttons, and executes real DOM browser workflows.

### 3. Independent Grader & Anti-Gaming
- **`grader/grader.py`**: The objective evaluation engine. Completely decoupled from the agent.
  - **`evaluate_order(order, task_constraints)`**: The core dynamic grading function.
    1. Verifies order exists on server and status is `completed`.
    2. Recalculates total spend directly from locked `catalog.json` (completely immune to client-side price tampering).
    3. Checks required product IDs and categories.
    4. Evaluates attribute requirements (e.g. `connectivity == "Wireless"`, `color == "Black"`, `type == "Mechanical"`).
    5. Validates hardware compatibility: checks `CPU.socket == Motherboard.socket`.
    6. Verifies capacity thresholds: aggregated RAM $\ge 16\text{GB}$, storage $\ge 500\text{GB}$.
    7. Checks budget constraint.
    8. Computes normalized WebShop score $S \in [0.0, 1.0]$.
  - **`grade_task(task_id, order_id)`**: command-line utility to grade any order by ID.

### 4. Agents
- **`agents/scripted_agent.py`**: Rule-based baseline agent that solves all 5 tasks to establish ground-truth metrics. Saves a report to `reports/runs/` after every episode.
- **`agents/browser_agent.py`**: Unified Playwright Chromium agent that supports both scripted and Groq LLM modes in headed (`--headed`) or headless mode. Automatically opens the browser, interacts with the DOM, extracts confirmed order IDs, and auto-saves run reports.
- **`agents/ppo_agent.py`**:
  - `DiscreteShoppingWrapper`: flattens action space for standard RL algorithms.
  - `LightweightPolicyGradientAgent`: zero-dependency pure NumPy Policy Gradient (REINFORCE with Baseline) that trains on CPU in seconds.
  - `train_sb3_ppo()`: drop-in function to train Stable-Baselines3 PPO when PyTorch is installed.
  - Saves a training summary report to `reports/runs/` after training finishes.
- **`agents/groq_agent.py`**: Autonomous LLM agent powered by Groq Cloud API (e.g. `qwen/qwen3.8-27b`). Formats observations, reasons about constraints, and outputs strict JSON actions. Saves a report to `reports/runs/` after every episode.

### 5. Run Reporter
- **`reports/run_reporter.py`**: Shared module called by all agents after every episode.
  - **`save_run_report(result, agent_type, model)`**: grades the episode's order, then writes two files to `reports/runs/`:
    - `YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.json` — full machine-readable record.
    - `YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.md` — human-readable Markdown summary.
  - Each report contains: timestamp, task id/name, agent type, model, steps, total reward, real cart total (recalculated from catalog), WebShop score, pass/fail, constraint check breakdown, and order id.
  - The grader is called **independently** inside the reporter — the agent cannot influence the graded output.

### 6. Test Suite & Exploit Verification
- **`tests/test_env.py`**: unit tests verifying Gymnasium API compliance, observation bounds, action processing, and episode truncation.
- **`tests/test_grader.py`**: unit tests verifying all 5 tasks against both valid orders (must score 1.0) and invalid/over-budget orders (must fail).
- **`tests/test_exploits.py`**: unit tests mounting two deliberate adversarial reward hacking attempts:
  - *Exploit 1:* Hollow checkout with empty cart / unconfirmed order. (Blocked: score = 0.0)
  - *Exploit 2:* Client-side cart price tampering. (Blocked: grader recalculates real prices from catalog and rejects budget breach).

---

## 🎯 The Five Benchmark Tasks

| ID | Difficulty | Name | Max Steps | Budget | Key Challenge |
|:---:|:---:|:---|:---:|:---:|:---|
| **1** | Trivial | Direct Item Purchase | 10 | ₹1,000 | Find and buy specific bottle `BOT-01`. Direct add → checkout. |
| **2** | Easy | Constrained Wireless Mouse | 15 | ₹5,000 | Filter by attribute (`Wireless`), color (`Black`), and budget. Must avoid `MOU-03` (₹8,999 over budget). |
| **3** | Medium | Mech Keyboard & Mouse Bundle | 20 | ₹5,000 | Multi-item composition. Must pick wired mechanical KB (`KEY-02`) + wired mouse (`MOU-04`) within ₹5,000 total. |
| **4** | Hard | Socket-Compatible PC Core | 25 | ₹45,000 | Domain compatibility. CPU (`CPU-01`, LGA1700) must match Motherboard socket (`MB-01`, LGA1700) + 16GB RAM. |
| **5** | Very Hard | Complete Gaming PC Setup | 50 | ₹1,15,000 | 10-item long-horizon planning. CPU, MB, RAM, SSD, GPU, Case, PSU, Monitor, KB, Mouse all compatible under ₹1.15L. |

---

## 🏆 Research-Backed Reward Function

The reward formulation is grounded in the **Princeton WebShop benchmark** (*Yao et al., NeurIPS 2022*):

1. **Step Cost (Efficiency Penalty):**
   $$r_{\text{step}} = -0.01$$
   Penalizes wandering and encourages minimal trajectory length.
2. **Intermediate Grounded Attribute Shaping:**
   Adding an item that satisfies task target categories or product IDs provides $+0.1$ to $+0.25$ intermediate shaping reward to accelerate RL policy convergence.
3. **Terminal WebShop Score Formulation:**
   Upon checkout, the independent grader evaluates attribute overlap:
   $$r_{\text{terminal}} = 1.0 + \left( \frac{\sum \text{Satisfied Constraints}}{\text{Total Constraints}} \times 0.5 \right) \quad \text{if all passed}$$
   If constraints fail or budget is breached:
   $$r_{\text{terminal}} = (\text{Attribute Overlap} \times 0.3) - 0.2$$

---

## 🛡️ Anti-Gaming & Tamper Resistance

The environment is engineered specifically to prevent **reward hacking**:
1. **Zero Client Trust:** The grader never inspects the agent's stdout, conversation messages, or client-side JSON totals.
2. **Authoritative Server Verification:** The grader only queries the server's persisted `orders.json` and reconstructs line item costs from the immutable `catalog.json`.
3. **Automated Exploit Tests:** Running `pytest tests/test_exploits.py -v` demonstrates both exploit attacks failing with zero score.

---

## 🤖 AI Tools Used Disclosure

In accordance with the assignment ground rules:
- **Claude / Gemini (via Antigravity IDE):** Used for codebase scaffolding, architectural structuring, and documentation drafting.
- **Pytest & Gymnasium:** Standard open-source tooling for environment compliance and automated unit testing.
- **Groq API:** Cloud inference API for LLM-as-policy evaluation.

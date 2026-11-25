# 🛒 Shopping RL Environment (`shopping-rl-env`)

> A Farama Gymnasium-compliant reinforcement learning and evaluation environment for autonomous shopping agents under multi-attribute and compositional constraints. Grounded in the Princeton WebShop formulation with anti-gaming independent verification.

---

## ⚡ Quickstart — Two Ways to Run

You can run the environment either **with Docker** (fastest, zero local Python setup) or **locally via `venv`** (needed if you want to watch Chromium run visually with `--headed`).

---

### Option A: Run with Docker (Recommended — Linux, macOS, Windows)
Docker provides a self-contained, reproducible headless environment across all operating systems without local Python or Playwright setup:

```bash
# Optional: only needed if running Groq LLM agents
cp .env.example .env   # then add your free-tier GROQ_API_KEY

# 1. Build and start the store + run baseline agent on all 5 tasks:
docker compose up --build

# 2. Run the full test suite (20 tests) inside Docker:
docker compose run --rm agent-eval pytest tests/ -v

# 3. Run adversarial exploit verification inside Docker:
docker compose run --rm agent-eval python tests/test_exploits.py

# Or run inside the running store container:
docker exec -it shopping-rl-store python tests/test_exploits.py

# 4. Run Policy Gradient or Groq LLM agent inside Docker:
docker compose run --rm agent-eval python agents/ppo_agent.py 1
docker compose run --rm agent-eval python agents/groq_agent.py 1 qwen/qwen3.8-27b

# 5. Grade a saved store order by ID:
docker compose run --rm agent-eval python -m grader.grader 1 <ORDER_ID>
```

---

### Option B: Run Locally with Python `venv` (Required for `--headed` Browser Display)

#### 1. Setup Virtual Environment & Optional `.env` (Linux, macOS & Windows)

##### On Linux / macOS:
```bash
# Skip `cd shopping-rl-env` if you are already inside the repo root
python3 -m venv .venv

# Activate venv (choose your shell):
source .venv/bin/activate        # Bash / Zsh
source .venv/bin/activate.fish   # Fish shell

pip install -r requirements.txt
playwright install chromium
```

> **Tip (Any shell, no activation required):** Commands below use `./.venv/bin/python3` directly on Linux/macOS so they work out of the box in Bash, Zsh, and Fish even without activating `.venv` (once activated — or on Windows — you can use `python` and `pytest` instead).

##### Optional (only needed for `--mode groq` / `agents/groq_agent.py`):
```bash
cp .env.example .env
# Edit .env and add your free-tier GROQ_API_KEY
```

##### On Windows (PowerShell):
```powershell
cd shopping-rl-env
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
$env:PYTHONPATH="."
```

##### On Windows (Command Prompt / CMD):
```cmd
cd shopping-rl-env
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
playwright install chromium
set PYTHONPATH=.
```

#### 2. Run Baseline Agent (All 5 Tasks)
Evaluates the deterministic baseline policy across all 5 benchmark tasks and auto-saves run reports:
```bash
PYTHONPATH=. ./.venv/bin/python3 agents/scripted_agent.py
```
*(On Windows CMD: `set PYTHONPATH=. && python agents/scripted_agent.py`)*

#### 3. Run Headed Chromium Browser Agent (Native Host Display)
> **Note:** Because Docker runs in a headless container without direct display access on macOS and Windows, visual browser execution runs natively on your host OS:

```bash
# 1. Run Scripted Agent in visible Chromium (Task 1)
PYTHONPATH=. ./.venv/bin/python3 agents/browser_agent.py --task 1 --mode scripted --headed

# 2. Run Scripted Agent on Task 2 (Constrained Wireless Mouse)
PYTHONPATH=. ./.venv/bin/python3 agents/browser_agent.py --task 2 --mode scripted --headed

# 3. Run Groq LLM Agent in visible Chromium (AI reads screen & drives browser)
PYTHONPATH=. ./.venv/bin/python3 agents/browser_agent.py --task 1 --mode groq --headed

# (Omit --headed to run in headless background mode)
```

#### 4. Run Fast Headless RL Training & Evaluation
```bash
# Groq LLM in Gymnasium environment
PYTHONPATH=. ./.venv/bin/python3 agents/groq_agent.py 1 qwen/qwen3.8-27b

# Pure NumPy Policy Gradient RL trainer
PYTHONPATH=. ./.venv/bin/python3 agents/ppo_agent.py 1
```

#### 5. Run Full Test Suite (20 Tests)
```bash
PYTHONPATH=. ./.venv/bin/python3 -m pytest tests/ -v
```

#### 6. Run Adversarial Exploit Tests (Anti-Gaming Proof)
```bash
PYTHONPATH=. ./.venv/bin/python3 tests/test_exploits.py
```

#### 7. Grade a Saved Store Order by ID
Grade any persisted order in `store/orders.json` against a benchmark task using the independent grader CLI:
```bash
PYTHONPATH=. ./.venv/bin/python3 -m grader.grader 1 <ORDER_ID>
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
    - `_add_item_to_cart()`: validates product ID, positive quantity, and prevents cart quantities from exceeding available `stock`.
    - `_process_checkout()` (shared by `api_checkout()` and `ui_checkout()`): validates live inventory at checkout time, rejects over-stock carts (`400`), decrements in-memory stock, calculates real totals, assigns an order ID, and commits to `orders.json`.
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
    2. Recalculates total spend directly from locked `catalog.json` (completely immune to client-side price tampering) and validates positive quantities against available `stock`.
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
    - `YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.json` — full machine-readable record including task instruction and step-by-step trajectory (`prompt`, `thought`, `raw_response`, `obs`, `next_obs`, `state`, `action`, `reward`, `cumulative_reward`).
    - `YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.md` — human-readable Markdown summary with constraint checks and step-by-step trace.
  - Each report contains: timestamp, task id/name/instruction, agent type, model, steps, total reward, real cart total (recalculated from catalog), WebShop score, pass/fail, constraint check breakdown, order id/details, and trajectory steps.
  - The grader is called **independently** inside the reporter — the agent cannot influence the graded output.

### 6. Test Suite & Exploit Verification
- **`tests/test_env.py`**: unit tests verifying Gymnasium API compliance, observation bounds, action processing, stock limits, one-time shaping bounds, and episode truncation.
- **`tests/test_grader.py`**: unit tests verifying all 5 tasks against valid orders, invalid/over-budget orders, and stock/quantity limits.
- **`tests/test_store.py`**: unit tests verifying Flask REST API and HTML UI stock validation on cart additions, checkout-time inventory enforcement across concurrent sessions, and out-of-stock UI state.
- **`tests/test_exploits.py`**: unit tests mounting three deliberate adversarial reward hacking attempts:
  - *Exploit 1:* Hollow checkout with empty cart / unconfirmed order. (Blocked: score = 0.0)
  - *Exploit 2:* Client-side cart price tampering. (Blocked: grader recalculates real prices from catalog and rejects budget breach).
  - *Exploit 3:* Intermediate reward-shaping farming (`ADD_TO_CART` repeats and `ADD`/`REMOVE` cycles). (Blocked: one-time per-item shaping claim caps cumulative return strictly below optimal `1.74`).

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

## 🛠️ Tooling, AI Assistance & Data Provenance

- **Core Stack:** Farama Gymnasium (`gymnasium`), `pytest`, `numpy`, Playwright Chromium, and Flask.
- **LLM Policy Inference:** Groq Cloud free-tier inference (`qwen/qwen3.8-27b`) in `agents/groq_agent.py` and `agents/browser_agent.py` (`--mode groq`).
- **AI Coding Assistance:** Antigravity AI coding assistants were used to help scaffold store UI templates, documentation drafts, and test boilerplate; all grading logic, stock/anti-gaming defenses, bug fixes, and verification runs were reviewed and validated against `pytest` and the exploit suite.
- **Data Provenance:** All 32 product entries in `store/catalog.json` and the 5 declarative tasks in `environment/tasks.json` are self-contained synthetic benchmark data created specifically for this environment.

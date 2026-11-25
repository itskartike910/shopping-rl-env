# Environment Results & Evaluation Note

**Project:** Shopping RL Environment (`shopping-rl-env`)  
**Specification:** Verified Benchmarks, Exploit Verification & Honest Limitations  

---

## 1. Verified Agent Runs

### A. Scripted Baseline (`agents/scripted_agent.py` on `ShoppingEnv`)
Deterministic rule-based baseline run across all 5 tasks (post-fix verified output):

| Task ID | Task Difficulty | Objective | Steps Taken | Total Reward | Total Real Spend | WebShop Score | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Task 1** | Trivial | Buy Milton Steel Bottle (`BOT-01`) | 2 | 1.740 | ₹599 / ₹1,000 | 1.000 | **PASSED** |
| **Task 2** | Easy | Black wireless mouse under ₹5,000 (`MOU-01`) | 2 | 1.590 | ₹2,999 / ₹5,000 | 1.000 | **PASSED** |
| **Task 3** | Medium | Wired mech KB + wired mouse (`KEY-02` + `MOU-04`) | 3 | 1.680 | ₹3,598 / ₹5,000 | 1.000 | **PASSED** |
| **Task 4** | Hard | Compatible CPU + MB + 16GB RAM (`CPU-01`, `MB-01`, `RAM-02`) | 4 | 1.770 | ₹38,797 / ₹45,000 | 1.000 | **PASSED** |
| **Task 5** | Very Hard | Complete 10-item Gaming PC Setup | 11 | 2.400 | ₹107,390 / ₹115,000 | 1.000 | **PASSED** |

### B. Playwright Headless Browser Run (`agents/browser_agent.py`)
- Executed Task 1 in `--mode scripted` against Flask store (`--base-url http://127.0.0.1:5056`):
  - **Outcome:** **PASSED** (`3` browser steps: `SEARCH 'bottle'` → `ADD_TO_CART 'BOT-01'` → `CHECKOUT`, Order ID `ORD-E62A793B`, Real Total `₹599`, WebShop Score `1.000`, Total Reward `1.500`).
  - *Scope note:* Tasks 2–5 and `--mode groq` were not benchmarked in that headless run.

### C. Incomplete Policy Gradient Training Experiment (`agents/ppo_agent.py`)
- Executed `LightweightPolicyGradientAgent` (pure NumPy 2-layer REINFORCE with Baseline, `Discrete(33)` wrapper, 250 unseeded episodes) on Task 1:
  - **Training Curve:** Episodes 1–50 average reward `-0.005`; Episodes 201–250 average reward `+0.013` (last episode `+0.000`).
  - **Post-Training Greedy Evaluation:** **FAILED** (`10` steps, `total_reward = -0.1000`, `truncated = True`, `"No order placed — episode ended without checkout"`).
  - **Honest Interpretation:** This 250-episode CPU run is an incomplete training smoke test only — it verifies the gradient update loop and greedy evaluation reporting, **not** task mastery or RL convergence.

---

## 2. Adversarial Exploit Attempts (Anti-Gaming Analysis)

To verify resilience against reward hacking, three deliberate adversarial exploit suites (`tests/test_exploits.py`) plus stock/quantity checks (`tests/test_grader.py`, `tests/test_store.py`) were executed (**20/20 pytest tests passed**):

### Exploit 1: Empty / Hollow Checkout Claim
- **Attack Vector:** An agent executes `CHECKOUT` with `items: []` (or `status: "not_placed"`), claiming completion without ordering.
- **Grader Defense:** `evaluate_order()` checks `order.get("status") == "completed"` and non-empty `order.get("items")`.
- **Result:** **BLOCKED** (`passed: False`, `webshop_score: 0.0`, reason: *"Order status is not 'completed' or items list is empty"*).

### Exploit 2: Client-Side Cart Price & Budget Tampering
- **Attack Vector:** On Task 2 (₹5,000 budget), an agent orders `GPU-01` (RTX 4060, catalog price ₹34,999) while forging `total: 2999` and item `price: 2999`.
- **Grader Defense:** `evaluate_order()` ignores client-supplied prices/totals, looks up locked `catalog.json` prices by product ID, and enforces positive quantities aggregated against `stock`.
- **Result:** **BLOCKED** (`passed: False`, `real_total: 34999`, `budget_constraint: False`, `webshop_score: 0.0`).

### Exploit 3: Intermediate Reward-Shaping Farming
- **Attack Vector:** An RL agent repeatedly calls `ADD_TO_CART BOT-01` (9×) or cycles `ADD_TO_CART` / `REMOVE_FROM_CART` to collect `+0.25` shaping bonuses without efficiently solving Task 1.
- **Environment Defense:** `ShoppingEnv.step()` tracks `shaping_claimed` per episode so each required product ID / category shaping bonus is awarded at most once per episode.
- **Result:** **BLOCKED** (`9x ADD_TO_CART + CHECKOUT` yields cumulative return `0.0200`, and 4× `ADD`/`REMOVE` cycles yield `1.6600`, both strictly below optimal `1.7400`).

---

## 3. What These Runs Do NOT Prove
1. **They do not prove generalized open-web agency:** Real-world e-commerce websites involve dynamic DOMs, anti-bot protections, CAPTCHAs, authentication, and non-deterministic inventory.
2. **They do not prove RL policy mastery or multi-task generalization:** Scripted runs validate transitions and the grader; the 250-episode NumPy PG experiment failed greedy evaluation (`-0.1000`).
3. **In-process Python vs container boundary:** When `ShoppingEnv` runs in the same Python process as an agent script, process-level memory isolation relies on running via the separate Flask service / Docker containers (`shopping-store` vs `agent-eval`).
4. **Local Docker daemon check:** Docker CLI/Compose files are present, but Docker daemon execution should be verified on a host with `/var/run/docker.sock` available.

---

## 4. How to Measure Whether Training Improves a Model
To evaluate whether RL or post-training on this environment produces real capability gains:
1. **Greedy Evaluation Pass Rate & Learning Curves:** Run multi-seed training (5,000–10,000+ episodes) with Stable-Baselines3 PPO or GRPO and measure deterministic post-training pass rates rather than training-loop averages.
2. **Held-Out Catalog Generalization:** Evaluate trained models zero-shot on unseen product names, prices, sockets, and attribute combinations.
3. **Trajectory Step Efficiency:** Measure reduction in redundant exploration steps compared to untrained LLM/RL baselines.
4. **Constraint Adherence Rate:** Track the fraction of rollouts avoiding budget overflow, stock breaches, and socket incompatibilities.

# Shopping RL Environment Architecture

## 1. System Topology

The repository supports two execution paths backed by the same locked catalog (`store/catalog.json`), declarative task specs (`environment/tasks.json`), independent grader (`grader/grader.py`), and trajectory reporter (`reports/run_reporter.py`):

```
+-----------------------------------------------------------------------------+
|                          SHOPPING RL ENVIRONMENT                            |
|                                                                             |
|  PATH A: IN-MEMORY GYMNASIUM ENV         PATH B: FLASK WEB STORE + BROWSER  |
|  +------------------------------+        +-------------------------------+  |
|  |  agents/scripted_agent.py    |        |  agents/browser_agent.py      |  |
|  |  agents/ppo_agent.py         |        |  (Playwright Chromium DOM)    |  |
|  |  agents/groq_agent.py        |        +---------------+---------------+  |
|  +--------------+---------------+                        | HTTP / HTML      |
|                 | reset() / step()                       v                  |
|                 v                        +-------------------------------+  |
|  +------------------------------+        |  store/app.py (Flask :5000)   |  |
|  |  environment/env.py          |        |  - /api/* & HTML templates/   |  |
|  |  (ShoppingEnv - Pure Python) |        |  - Live stock decrement       |  |
|  |  - MultiDiscrete([8, 37])    |        |  - Writes ORD-* -> orders.json|  |
|  |  - Box (21-dim normalized)   |        +---------------+---------------+  |
|  |  - One-time shaping per item |                        |                  |
|  |  - In-memory ORD-ENV-*       |                        |                  |
|  +--------------+---------------+                        |                  |
|                 |                                        |                  |
|                 +------------------+---------------------+                  |
|                                    | Order Record                           |
|                                    v                                        |
|  +-----------------------------------------------------------------------+  |
|  |  INDEPENDENT GRADER (grader/grader.py)                                |  |
|  |  - Reads locked store/catalog.json & environment/tasks.json           |  |
|  |  - Recalculates real total; validates 0 < qty <= stock & constraints  |  |
|  +---------------------------------+-------------------------------------+  |
|                                    |                                        |
|                                    v                                        |
|  +-----------------------------------------------------------------------+  |
|  |  RUN REPORTER (reports/run_reporter.py -> reports/runs/)              |  |
|  |  - Logs JSON + Markdown: instruction, prompt, steps, obs, thoughts,   |  |
|  |    order breakdown, rewards, and independent grader verification      |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
```

## 2. Component Directory Structure

```
shopping-rl-env/
├── store/
│   ├── app.py                # Flask web store, REST API, and checkout stock verification
│   ├── catalog.json          # Locked catalog of 32 products (single source of truth)
│   ├── orders.json           # Authoritative server-side orders database (browser/API path)
│   └── templates/            # HTML templates for browser interaction
│       ├── layout.html       # Base responsive UI layout + flash alert banner
│       ├── index.html        # Main store front with search, sort, filter, stock state
│       ├── product.html      # Product specs, technical details, and stock indicator
│       ├── cart.html         # Cart breakdown, stock warnings, and checkout error alert
│       └── checkout.html     # Confirmed order details
│
├── environment/
│   ├── env.py                # In-memory Gymnasium ShoppingEnv with WebShop reward
│   ├── tasks.json            # 5 declarative task definitions
│   └── browser_env.py        # Minimal Playwright Chromium DOM driver
│
├── grader/
│   └── grader.py             # Schema-driven independent verifier
│
├── agents/
│   ├── scripted_agent.py     # Deterministic baseline agent (saves full trace report)
│   ├── ppo_agent.py          # NumPy Policy Gradient & SB3 PPO trainer + greedy eval
│   ├── groq_agent.py         # Autonomous Groq LLM agent (saves prompt & trace report)
│   └── browser_agent.py      # Unified Playwright Chromium agent (scripted/groq, headed/headless)
│
├── tests/
│   ├── test_env.py           # Gymnasium API, state transitions, stock & shaping tests
│   ├── test_grader.py        # Task constraint, stock & quantity grading tests
│   ├── test_store.py         # Flask API/UI stock & checkout enforcement tests
│   └── test_exploits.py      # Adversarial reward hacking & shaping-farm exploit tests
│
├── reports/
│   ├── run_reporter.py       # Shared reporter — saves full trace JSON + Markdown per run
│   ├── runs/                 # Auto-created run reports (YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.*)
│   ├── scope.md              # Half-page environment scope note
│   └── results.md            # Verified benchmark results, exploits, and limitation note
│
├── docs/
│   └── architecture.md       # Architectural specification and data flow
│
├── Dockerfile                # Reproducible container configuration
├── docker-compose.yml        # Docker compose service definition
├── requirements.txt          # Python dependencies
└── README.md                 # Setup, run/grade commands, AI tooling, and file descriptions
```

## 3. Data Flow & Security Boundaries

1. **State Isolation & Scope:** In Path A (`ShoppingEnv`), agents receive `obs` and `info` from `env.step()` and orders (`ORD-ENV-*`) are evaluated in memory by `evaluate_order()`. Note that an in-process Python script shares address space with `ShoppingEnv`; stronger isolation is achieved in Path B or Docker (`shopping-store` vs `agent-eval` containers), where `orders.json` and `catalog.json` reside on the server side.
2. **Authoritative Valuation, Stock & Shaping Anti-Gaming:** Any client-reported price or total is ignored. `grader.py` always computes `sum(catalog_by_id[item.id].price * qty)` and enforces `0 < qty <= stock` aggregated per product ID against `catalog.json`. In `ShoppingEnv.step()`, intermediate attribute-shaping bonuses (`+0.25` / `+0.10`) are granted at most once per unique required product ID / category per episode (`shaping_claimed`), blocking `ADD_TO_CART` / `REMOVE_FROM_CART` reward farming.
3. **Run Reports:** After every episode, `reports/run_reporter.py` independently grades the order and writes full JSON + Markdown trace reports (task instruction, prompts, observations, thoughts, actions, step rewards, and constraint checks) to `reports/runs/`.
4. **Execution Parity:** Both Path A (Gym) and Path B (Flask + Playwright) enforce identical catalog pricing, search/filter semantics, positive stock bounds, and `evaluate_order()` task constraint checks.

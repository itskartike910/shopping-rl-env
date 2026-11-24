# Shopping RL Environment Architecture

## 1. System Topology

```
+-------------------------------------------------------------------------+
|                        SHOPPING RL ENVIRONMENT                          |
|                                                                         |
|  +--------------------+         +------------------------------------+  |
|  |    AGENT LAYER     | actions |         LOCAL FLASK STORE          |  |
|  |                    |-------->|          (localhost:5000)          |  |
|  | 1. Scripted Base   |<--------|  - REST API (/search, /cart...)    |  |
|  | 2. Policy Gradient | obs/rew |  - HTML UI (Responsive Web Store)  |  |
|  | 3. Groq LLM Agent  |         +-----------------+------------------+  |
|  +--------------------+                           |                     |
|            |                                      | Order Record        |
|            v (after every episode)                v (orders.json)       |
|  +--------------------+                 +----------------------------+  |
|  |   RUN REPORTER     |                 |    INDEPENDENT GRADER      |  |
|  |  reports/runs/     |                 |  (grader/grader.py)        |  |
|  |  - JSON record     |                 |  - Reads locked catalog    |  |
|  |  - Markdown report |                 |  - Recalculates real total |  |
|  +--------------------+                 |  - Dynamic Schema Checker  |  |
|                                         +----------------------------+  |
|  +--------------------+                           |                     |
|  |   GYMNASIUM ENV    |                           |                     |
|  |  ShoppingEnv       |<--------------------------|                     |
|  |  - Farama API      |     verified score        |                     |
|  |  - MultiDiscrete   |                           |                     |
|  |  - Box (21-dim)    |                           |                     |
|  |  - WebShop Reward  |                           |                     |
|  +--------------------+                           |                     |
+-------------------------------------------------------------------------+
```

## 2. Component Directory Structure

```
shopping-rl-env/
├── store/
│   ├── app.py                # Flask web store and API service
│   ├── catalog.json          # Locked catalog of 32 products (single source of truth)
│   ├── orders.json           # Authoritative server-side orders database
│   └── templates/            # HTML templates for browser interaction
│       ├── layout.html       # Base responsive UI layout
│       ├── index.html        # Main store front with search, sort, filter
│       ├── product.html      # Product specs and technical details
│       ├── cart.html         # Cart breakdown and checkout
│       └── checkout.html     # Confirmed order details
│
├── environment/
│   ├── env.py                # Gymnasium ShoppingEnv with WebShop reward
│   ├── tasks.json            # 5 declarative task definitions
│   └── browser_env.py        # Playwright Chromium browser driver
│
├── grader/
│   └── grader.py             # Schema-driven independent verifier
│
├── agents/
│   ├── scripted_agent.py     # Deterministic baseline agent (saves report after run)
│   ├── ppo_agent.py          # Policy Gradient & Stable-Baselines3 PPO trainer (saves report)
│   ├── groq_agent.py         # Autonomous Groq LLM agent (saves report after run)
│   └── browser_agent.py      # Unified Playwright Chromium agent (scripted/groq, headed/headless)
│
├── tests/
│   ├── test_env.py           # Gymnasium API & state transition tests
│   ├── test_grader.py        # Task constraint grading verification tests
│   └── test_exploits.py      # Adversarial reward hacking exploit tests
│
├── reports/
│   ├── run_reporter.py       # Shared reporter — saves JSON + Markdown after each episode
│   ├── runs/                 # Auto-created run reports (YYYY-MM-DD_HH-MM-SS_task{N}_{agent}.*)
│   ├── scope.md              # Half-page environment scope note
│   └── results.md            # Results, exploit outcomes, and limitation note
│
├── docs/
│   └── architecture.md       # Architectural specification and data flow
│
├── Dockerfile                # Reproducible container configuration
├── docker-compose.yml        # Docker compose service definition
├── requirements.txt          # Python dependencies
└── README.md                 # Complete guide, quickstart, and file descriptions
```

## 3. Data Flow & Security Boundaries

1. **State Isolation:** The agent only interacts with observations returned by `env.step()`. It never has direct write access to `catalog.json` or `orders.json`.
2. **Authoritative Valuation:** Any price or discount reported by the agent or client session is ignored. The grader always performs `sum(catalog_by_id[item.id].price * qty)`.
3. **Run Reports:** After every episode, `reports/run_reporter.py` calls the grader independently and writes a tamper-proof record to `reports/runs/`. The agent cannot modify this output.
4. **Execution Parity:** Both the direct API (Gym) and visual Playwright Chromium paths manipulate the same state model, ensuring test parity.

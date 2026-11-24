# Environment Results & Evaluation Note

**Project:** Shopping RL Environment (`shopping-rl-env`)  
**Specification:** Baseline Benchmarks & Exploit Verification  

---

## 1. Baseline Agent Runs Across All 5 Tasks

The deterministic baseline agent (`agents/scripted_agent.py`) and Policy Gradient agent (`agents/ppo_agent.py`) were evaluated on the 5 benchmark tasks.

| Task ID | Task Difficulty | Objective | Steps Taken | Total Real Spend | WebShop Score | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **Task 1** | Trivial | Buy Milton Steel Bottle (`BOT-01`) | 2 | ₹599 / ₹1,000 | 1.000 | **PASSED** |
| **Task 2** | Easy | Black wireless mouse under ₹5,000 (`MOU-01`) | 2 | ₹2,999 / ₹5,000 | 1.000 | **PASSED** |
| **Task 3** | Medium | Wired mech KB + wired mouse (`KEY-02` + `MOU-04`) | 3 | ₹3,598 / ₹5,000 | 1.000 | **PASSED** |
| **Task 4** | Hard | Compatible CPU + MB + 16GB RAM (`CPU-01`, `MB-01`, `RAM-02`) | 4 | ₹38,797 / ₹45,000 | 1.000 | **PASSED** |
| **Task 5** | Very Hard | Complete 10-item Gaming PC Setup | 11 | ₹107,390 / ₹115,000 | 1.000 | **PASSED** |

### Policy Gradient RL Training Outcome:
On Task 1, training the built-in Policy Gradient agent over 250 episodes demonstrated clear policy improvement:
- **Initial Episodes (1–50):** Average return of `-0.100` (episodes timing out without valid checkout).
- **Final Episodes (200–250):** Average return improved to `+0.034` (with individual trajectory returns reaching `+0.150` to `+1.490` as the agent learned to locate and add `BOT-01`).

---

## 2. Adversarial Exploit Attempts (Anti-Gaming Analysis)

To test resilience against reward hacking, two deliberate adversarial exploits were mounted against the environment grader (`tests/test_exploits.py`):

### Exploit 1: Empty / Hollow Checkout Claim
- **Attack Vector:** An agent immediately executes the `CHECKOUT` action with zero items in the cart, or sends an unfinalized order status payload claiming *"I successfully bought the product"*.
- **Vulnerability in Weak Environments:** In naive or regex-based environments, reaching a terminal "order page" or outputting a success keyword triggers a positive reward signal.
- **Grader Defense:** `evaluate_order()` inspects the authoritative server database. It validates `status == "completed"` and checks `len(order["items"]) > 0`.
- **Result:** **BLOCKED**. The grader returns `passed: False`, `webshop_score: 0.0`, reason: *"Order status is not 'completed' or items list is empty"*.

### Exploit 2: Client-Side Cart Price Tampering
- **Attack Vector:** An agent attempts Task 2 (budget ₹5,000). It adds an out-of-budget flagship component (e.g., `GPU-01`, RTX 4060 priced at ₹34,999), but modifies the client-side session payload to report `total: 2999` and item price `2999`.
- **Vulnerability in Weak Environments:** Environments that trust client/agent-reported dictionaries or API query parameters accept the fabricated total.
- **Grader Defense:** The grader discards any client-supplied totals. It extracts only the product IDs and looks up authoritative prices directly in the locked, server-side `catalog.json`. The recalculated price of ₹34,999 violates the ₹5,000 budget constraint.
- **Result:** **BLOCKED**. The grader returns `passed: False`, `real_total: 34999`, reason: *"Total price ₹34,999 exceeds budget ₹5,000"*.

---

## 3. What These Runs Do NOT Prove
Honest reporting requires acknowledging what current runs cannot demonstrate:
1. **They do not prove generalized open-web agency:** Passing tasks in this simulated environment does not mean an agent can navigate arbitrary real-world e-commerce websites with dynamic anti-bot protections, captchas, or infinite DOM variations.
2. **They do not prove zero-shot multi-task generalization:** A policy trained specifically on Task 1 will not automatically generalize to the 10-step compositional reasoning required in Task 5 without curriculum training.
3. **Short sample size limitations:** A few successful seed runs demonstrate that the state transitions and verifiers are mathematically sound, but do not establish asymptotic convergence across high-variance exploration spaces.

---

## 4. How to Measure Whether Training Improves a Model
To evaluate whether RL fine-tuning on this environment produces real capability gains, a frontier lab should measure:
1. **Held-Out Catalog Generalization:** Evaluate the trained agent on an unseen catalog containing new product names, prices, and attribute tags. Measure the zero-shot task completion rate.
2. **Trajectory Step Efficiency (Length-Normalized Return):** Measure the reduction in step count required to complete a multi-constraint task. If an untrained LLM requires 18 exploratory steps and a fine-tuned model completes it in 4 direct actions without redundant queries, policy improvement is confirmed.
3. **Constraint Adherence Rate:** Track the percentage of episodes where the agent avoids budget overflow and incompatible hardware pairings (e.g., socket mismatch) during intermediate exploration.
4. **Reward Curve Asymptote:** Track normalized WebShop attribute overlap score curves across 10,000+ training episodes under multiple random seeds.

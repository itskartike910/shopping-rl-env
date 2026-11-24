# Environment Scope Note: Multi-Constraint E-Commerce Shopping RL Environment

**Project:** Shopping RL Environment (`shopping-rl-env`)  
**Role:** Lead RL Environment Engineer  
**Candidate Evaluation:** Adzzat Labs  

---

## 1. What the Agent Does
The agent operates inside an interactive e-commerce shopping environment featuring a catalog of 32 diverse technical products (CPUs, motherboards, RAM, SSDs, GPUs, PC cases, power supplies, monitors, keyboards, mice, audio equipment, and lifestyle accessories). 

Given a compositional natural language task instruction (e.g., *"Assemble a complete PC gaming build with compatible CPU and Motherboard sockets, at least 16GB RAM, and 500GB SSD under ₹1,15,000"*), the agent interacts step-by-step with the environment. It can perform discrete actions:
- **Search:** Query products by name, tag, or spec (e.g., `LGA1700`, `AM5`, `wireless`, `mechanical`).
- **Sort:** Dynamically order catalog items by price ascending/descending, customer rating, or name.
- **Filter:** Narrow viewports by product category.
- **Inspect:** View granular product specifications (socket type, switch color, wattage, RAM capacity).
- **Cart Management:** Add or remove items, review subtotal.
- **Checkout:** Commit the transaction, creating an immutable server-side order record.

The agent receives either a normalized 21-dimensional continuous vector observation (for reinforcement learning policies such as PPO) or a structured text/JSON state observation (for LLM-based autonomous agents).

---

## 2. What Success Means
Success is defined **strictly by verified ground-truth environment state**, never by the agent's self-reported completion claim or client-side cart totals:
1. **Order Finalization:** An authoritative order record must exist in the server database with `status: "completed"` and a non-empty item payload.
2. **Authoritative Price Verification:** Real purchase prices are recalculated directly from the immutable, server-side catalog (`catalog.json`). The order total must strictly satisfy the budget ceiling.
3. **Multi-Constraint Satisfaction:** The purchased items must satisfy all compositional task constraints:
   - Exact product identity match (Task 1).
   - Attribute specifications: connectivity (e.g., Wireless), switch type (Mechanical), color (Black) (Tasks 2 & 3).
   - Cross-item compatibility: CPU socket (`LGA1700` vs `AM5`) must physically match the Motherboard socket (Tasks 4 & 5).
   - Threshold capacities: aggregated RAM $\ge 16\text{ GB}$, storage $\ge 500\text{ GB}$ (Tasks 4 & 5).
   - Category completeness: all 10 hardware categories present in cart (Task 5).

A binary pass/fail score ($1.0$ or $0.0$) is returned alongside a continuous WebShop-style attribute overlap score $S \in [0.0, 1.0]$.

---

## 3. Why This Data is Useful to a Frontier AI Lab
Frontier AI labs training foundation models and browser-use agents face acute bottlenecks with real-world e-commerce data: live websites are non-deterministic, rate-limited, fragile, and prone to bot detection. 

This environment provides:
1. **Verifiable Agent Trajectories:** Complete, step-by-step action trajectories paired with definitive mathematical reward signals. Unlike code generation (where tests can have false positives) or open-ended web navigation (which requires noisy LLM-as-a-judge), this environment delivers 100% deterministic ground-truth verification.
2. **Compositional Reasoning & Planning Benchmarking:** Hard tasks (like Task 5) require long-horizon planning (up to 50 steps), multi-constraint budgeting, and domain compatibility rules (socket matching). This isolates whether a model can plan ahead without backtracking into budget overflow.
3. **Robust Anti-Gaming Training Data:** Because the independent verifier blocks hollow checkout exploits and price tampering, RL algorithms trained here cannot exploit shortcut loopholes (reward hacking), forcing policies to internalize true task semantics.
4. **Standardized Gymnasium Interface:** Drop-in compatibility with Farama Gymnasium and Stable-Baselines3 allows immediate RL fine-tuning (PPO, DPO, GRPO) and seamless transfer to larger web-agent benchmarks (e.g., WebShop, VisualWebArena).

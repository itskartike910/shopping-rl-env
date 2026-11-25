"""
Unit tests for the Gymnasium Shopping Environment.
Verifies API compliance, reset, step, observation spaces, and truncation.
"""

import pytest
import numpy as np
from environment.env import ShoppingEnv, ACTION_NAMES

def test_env_initialization():
    env = ShoppingEnv(task_id=1)
    assert env.action_space is not None
    assert env.observation_space is not None
    assert env.observation_space.shape == (21,)

def test_env_reset():
    env = ShoppingEnv(task_id=1)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (21,)
    assert info["task_id"] == 1
    assert info["steps_taken"] == 0
    assert len(info["cart"]) == 0

def test_env_search_action():
    env = ShoppingEnv(task_id=1)
    env.reset()
    # Search action (type 0, query index 1 = "cpu")
    obs, reward, terminated, truncated, info = env.step((0, 1))
    assert env.current_page == "search"
    assert env.steps_taken == 1
    assert not terminated

def test_env_sort_action():
    env = ShoppingEnv(task_id=2)
    env.reset()
    # Sort action (type 1, param 1 = "price_asc")
    obs, reward, terminated, truncated, info = env.step((1, 1))
    assert env.current_sort == "price_asc"
    assert env.steps_taken == 1

def test_env_step_limits():
    env = ShoppingEnv(task_id=1, max_steps=3)
    env.reset()
    env.step((0, 0))
    env.step((0, 0))
    obs, reward, terminated, truncated, info = env.step((0, 0))
    assert truncated is True
    assert terminated is False


def test_env_stock_limit_enforced():
    """MB-01 has stock=5; a 6th ADD_TO_CART must be rejected."""
    env = ShoppingEnv(task_id=4, max_steps=20)
    env.reset()
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step({"action": "ADD_TO_CART", "param": "MB-01"})
        assert obs[-1] == 1.0

    obs, reward, terminated, truncated, info = env.step({"action": "ADD_TO_CART", "param": "MB-01"})
    assert obs[-1] == 0.0
    assert info["cart"][0]["qty"] == 5


def test_env_shaping_reward_not_farmable():
    """Repeated ADD_TO_CART or ADD/REMOVE cycles cannot claim shaping bonus more than once."""
    env = ShoppingEnv(task_id=1, max_steps=15)
    env.reset()

    # First add of BOT-01 earns +0.25 - 0.01 = +0.24
    _, r1, _, _, _ = env.step({"action": "ADD_TO_CART", "param": "BOT-01"})
    assert round(r1, 4) == 0.24

    # Remove then re-add BOT-01 must NOT re-grant +0.25
    _, r_rem, _, _, _ = env.step({"action": "REMOVE_FROM_CART", "param": "BOT-01"})
    assert round(r_rem, 4) == -0.01

    _, r2, _, _, _ = env.step({"action": "ADD_TO_CART", "param": "BOT-01"})
    assert round(r2, 4) == -0.01

    # Optimal 2-step checkout stays 1.74; cycled trajectory must stay strictly lower
    _, r_chk, terminated, _, _ = env.step({"action": "CHECKOUT", "param": 0})
    assert terminated is True
    total = r1 + r_rem + r2 + r_chk
    assert round(total, 4) == 1.72
    assert total < 1.74

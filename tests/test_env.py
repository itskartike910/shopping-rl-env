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

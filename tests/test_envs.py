"""Môi trường học tăng cường (chương 35)."""

import numpy as np

from ml_course.envs import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    CartPole,
    CliffWalking,
    GridWorld,
    KArmedBandit,
    RandomWalk,
)


def test_gridworld_transitions_are_distributions():
    env = GridWorld("8x8", slip=0.2)
    for s in range(env.n_states):
        for a in range(4):
            assert abs(sum(p for p, *_ in env.P[s][a]) - 1) < 1e-12


def test_gridworld_deterministic_path_reaches_goal():
    env = GridWorld("4x4")
    env.reset(seed=0)
    rewards, done = [], False
    for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
        _, r, done = env.step(a)
        rewards.append(r)
    assert done and rewards[-1] == 1.0 and sum(rewards[:-1]) == 0


def test_gridworld_hole_terminates_without_reward():
    env = GridWorld("4x4")
    env.reset(seed=0)
    env.step(RIGHT)
    s, r, done = env.step(DOWN)                                         # (1, 1) là hố
    assert s == 5 and r == 0.0 and done


def test_cliffwalking_rewards():
    env = CliffWalking()
    env.reset(seed=0)
    s, r, done = env.step(RIGHT)                                        # rơi xuống vực
    assert (s, r, done) == (36, -100.0, False)
    s, r, done = env.step(UP)
    assert (s, r) == (24, -1.0)
    env.step(LEFT)


def test_random_walk_monte_carlo_matches_true_value():
    env = RandomWalk()
    env.reset(seed=0)
    returns = []
    for _ in range(4000):
        env.reset()
        done, reward = False, 0.0
        while not done:
            _, reward, done = env.step()
        returns.append(reward)
    assert abs(np.mean(returns) - env.true_values[2]) < 0.03


def test_bandit_reproducible():
    a, b = KArmedBandit(10, seed=3), KArmedBandit(10, seed=3)
    np.testing.assert_array_equal(a.q_star, b.q_star)
    assert [a.pull(1) for _ in range(5)] == [b.pull(1) for _ in range(5)]


def test_cartpole_constant_push_fails_quickly():
    env = CartPole()
    state = env.reset(seed=0)
    assert state.shape == (4,) and np.abs(state).max() <= 0.05
    steps, done = 0, False
    while not done:
        _, reward, done = env.step(1)
        steps += 1
        assert reward == 1.0
    assert steps < 100

"""Môi trường học tăng cường nhỏ, thuần NumPy (không cần gym) — cho chương 35.

Giao diện giống Gymnasium ở mức tối thiểu: ``reset(seed=None) -> state``, ``step(action) -> (state, reward, done)``.
Môi trường dạng bảng (``GridWorld``, ``CliffWalking``) còn có mô hình chuyển trạng thái đầy đủ
``P[s][a] = [(xác suất, s_tiếp, phần_thưởng, kết_thúc), ...]`` để dùng với quy hoạch động.
"""

from __future__ import annotations

import numpy as np

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = ("←", "↓", "→", "↑")
_MOVES = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"],
}


class KArmedBandit:
    """Bài toán k cần gạt: phần thưởng của cần gạt a ~ N(q*(a), 1), q*(a) ~ N(0, 1)."""

    def __init__(self, k: int = 10, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.q_star = self.rng.normal(0, 1, k)

    def pull(self, a: int) -> float:
        return float(self.rng.normal(self.q_star[a], 1))


class GridWorld:
    """Lưới kiểu FrozenLake: S xuất phát, F đi được, H hố (kết thúc, thưởng 0), G đích (kết thúc, thưởng 1).

    ``slip``: xác suất bị trượt sang một trong hai hướng **vuông góc** với hướng muốn đi (mỗi bên slip/2).
    ``step_reward``: phần thưởng mỗi bước không kết thúc (mặc định 0).
    """

    def __init__(self, layout: str | list[str] = "4x4", slip: float = 0.0, step_reward: float = 0.0, max_steps: int = 200):
        self.grid = np.array([list(row) for row in (MAPS[layout] if isinstance(layout, str) else layout)])
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states, self.n_actions = self.n_rows * self.n_cols, 4
        self.slip, self.step_reward, self.max_steps = slip, step_reward, max_steps
        self.start = int(np.flatnonzero(self.grid.ravel() == "S")[0])
        self.terminal = {s for s, c in enumerate(self.grid.ravel()) if c in "HG"}
        self.P = {s: {a: self._transitions(s, a) for a in range(4)} for s in range(self.n_states)}
        self.rng = np.random.default_rng()
        self.state, self.t = self.start, 0

    def _move(self, s, a):
        r, c = divmod(s, self.n_cols)
        dr, dc = _MOVES[a]
        r, c = min(max(r + dr, 0), self.n_rows - 1), min(max(c + dc, 0), self.n_cols - 1)
        return r * self.n_cols + c

    def _transitions(self, s, a):
        if s in self.terminal:
            return [(1.0, s, 0.0, True)]
        outcomes = {}
        for direction, p in [(a, 1 - self.slip), ((a - 1) % 4, self.slip / 2), ((a + 1) % 4, self.slip / 2)]:
            if p > 0:
                s2 = self._move(s, direction)
                outcomes[s2] = outcomes.get(s2, 0.0) + p
        result = []
        for s2, p in outcomes.items():
            cell = self.grid.ravel()[s2]
            reward = 1.0 if cell == "G" else (0.0 if cell == "H" else self.step_reward)
            result.append((p, s2, reward, cell in "HG"))
        return result

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state, self.t = self.start, 0
        return self.state

    def step(self, action: int):
        trans = self.P[self.state][action]
        i = self.rng.choice(len(trans), p=[p for p, *_ in trans])
        _, s2, reward, done = trans[i]
        self.state, self.t = s2, self.t + 1
        return s2, reward, done or self.t >= self.max_steps


class CliffWalking(GridWorld):
    """Lưới 4×12 (Sutton & Barto, ví dụ 6.6): mỗi bước −1; rơi xuống vực (hàng dưới, giữa S và G) −100 và quay về S."""

    def __init__(self, max_steps: int = 500):
        layout = ["." * 12, "." * 12, "." * 12, "S" + "C" * 10 + "G"]
        self.grid = np.array([list(row) for row in layout])
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states, self.n_actions = 48, 4
        self.slip, self.max_steps = 0.0, max_steps
        self.start, self.goal = 36, 47
        self.terminal = {self.goal}
        self.P = {s: {a: self._cliff_transitions(s, a) for a in range(4)} for s in range(48)}
        self.rng = np.random.default_rng()
        self.state, self.t = self.start, 0

    def _cliff_transitions(self, s, a):
        if s == self.goal:
            return [(1.0, s, 0.0, True)]
        s2 = self._move(s, a)
        if self.grid.ravel()[s2] == "C":
            return [(1.0, self.start, -100.0, False)]
        return [(1.0, s2, -1.0, s2 == self.goal)]


class RandomWalk:
    """Bước ngẫu nhiên 5 trạng thái A–E (Sutton & Barto, ví dụ 6.2): bắt đầu ở C, mỗi bước sang trái/phải với xác
    suất 1/2; thoát bên phải được thưởng 1, bên trái 0. Giá trị thật (γ = 1): 1/6, 2/6, ..., 5/6."""

    n_states = 5
    true_values = np.arange(1, 6) / 6

    def __init__(self):
        self.rng = np.random.default_rng()
        self.state = 2

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = 2
        return self.state

    def step(self, action=None):
        self.state += 1 if self.rng.random() < 0.5 else -1
        if self.state < 0:
            return None, 0.0, True
        if self.state > 4:
            return None, 1.0, True
        return self.state, 0.0, False


class CartPole:
    """Con lắc ngược trên xe (động lực học giống Gymnasium ``CartPole-v1``).

    Trạng thái: (vị trí xe, vận tốc xe, góc con lắc, vận tốc góc). Hành động: 0 = đẩy trái, 1 = đẩy phải.
    Thưởng +1 mỗi bước; kết thúc khi |góc| > 12° hoặc |vị trí| > 2.4; cắt ở ``max_steps`` (mặc định 500).
    """

    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    theta_threshold, x_threshold = 12 * 2 * np.pi / 360, 2.4
    n_actions, obs_dim = 2, 4

    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.state, self.t = None, 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state, self.t = self.rng.uniform(-0.05, 0.05, 4), 0
        return self.state.copy()

    def step(self, action: int):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.masscart + self.masspole
        polemass_length = self.masspole * self.length
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (self.length * (4.0 / 3.0 - self.masspole * cos**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * cos / total_mass
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc                 # Euler như Gymnasium
        theta, theta_dot = theta + self.tau * theta_dot, theta_dot + self.tau * theta_acc
        self.state, self.t = np.array([x, x_dot, theta, theta_dot]), self.t + 1
        failed = abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
        return self.state.copy(), 1.0, bool(failed or self.t >= self.max_steps)


__all__ = ["KArmedBandit", "GridWorld", "CliffWalking", "RandomWalk", "CartPole", "MAPS", "ACTION_NAMES",
           "LEFT", "DOWN", "RIGHT", "UP"]

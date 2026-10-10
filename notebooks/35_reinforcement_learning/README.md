# Chương 35 · Reinforcement Learning — Học tăng cường

> Học hành vi từ phần thưởng thay vì đáp án: bài toán k cần gạt và khám phá–khai thác, MDP và phương trình Bellman, quy hoạch động, Monte Carlo và TD, SARSA và Q-learning, rồi DQN và policy gradient với mạng nơ-ron, cuối cùng là thuật toán di truyền và neuroevolution — tối ưu không cần gradient. Mọi môi trường (FrozenLake, Cliff Walking, CartPole...) được viết sẵn bằng NumPy trong `ml_course.envs`, không cần cài gym.

## 🎯 Mục tiêu

- Bandit: tham lam vs ε-greedy vs UCB. MDP, return, γ; Bellman kỳ vọng (lặp và giải hệ tuyến tính) và tối ưu; value/policy iteration; hành vi tránh rủi ro trên băng trơn; ảnh hưởng của γ.
- MC vs TD(0) trên random walk; bias–variance của mục tiêu cập nhật; TD hội tụ về nghiệm Bellman.
- SARSA vs Q-learning (Cliff Walking: an toàn vs tối ưu); Q-learning trên môi trường ngẫu nhiên; thiên lệch cực đại và Double Q-learning.
- DQN (replay, target network, Double DQN) và sự bất ổn qua các seed; REINFORCE và baseline; actor-critic/PPO/RLHF.
- Thuật toán di truyền (bài toán cái túi, so với DP/tham lam/ngẫu nhiên), tỉ lệ đột biến; neuroevolution và Evolution Strategies cho CartPole.

## 📋 Yêu cầu trước

[Chương 2](../02_mathematics_for_ml/README.md) (xác suất, kỳ vọng), [Chương 27–28](../27_deep_learning_foundations/README.md) (mạng nơ-ron, PyTorch) cho bài 04.

## 🧰 Thư viện

NumPy; bài 04 dùng PyTorch (`pip install -e ".[dl]"`) trên CPU — mạng rất nhỏ nên CPU nhanh và tái lập được. Môi trường: `ml_course.envs` (`KArmedBandit`, `GridWorld`, `CliffWalking`, `RandomWalk`, `CartPole`).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — MDP, phương trình Bellman và quy hoạch động | 📖 Bài học | [View](01_mdp_and_bellman.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/01_mdp_and_bellman.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/01_mdp_and_bellman.ipynb) |
| 02 | Bài 02 — Monte Carlo và Temporal Difference | 📖 Bài học | [View](02_monte_carlo_and_td.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/02_monte_carlo_and_td.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/02_monte_carlo_and_td.ipynb) |
| 03 | Bài 03 — Q-learning và SARSA | 📖 Bài học | [View](03_q_learning_and_sarsa.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/03_q_learning_and_sarsa.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/03_q_learning_and_sarsa.ipynb) |
| 04 | Bài 04 — Deep Q-Network và Policy Gradient | 📖 Bài học | [View](04_dqn_and_policy_gradient.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/04_dqn_and_policy_gradient.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/04_dqn_and_policy_gradient.ipynb) |
| 05 | Bài 05 — Thuật toán di truyền và neuroevolution | 📖 Bài học | [View](05_neuroevolution_genetic_algorithm.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/05_neuroevolution_genetic_algorithm.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/05_neuroevolution_genetic_algorithm.ipynb) |
| 90 | Bài tập — Chương 35: Reinforcement Learning (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 35 · Reinforcement Learning | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/35_reinforcement_learning/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `run_bandit`, `evaluate_policy_iterative`, `value_iteration`, `policy_iteration`. Bài 02: `run_random_walk` (MC, TD(0)). Bài 03: `td_control` (SARSA, Q-learning), `maximization_bias_experiment`. Bài 04: `ReplayBuffer`, `train_dqn`, `train_reinforce`. Bài 05: `genetic_algorithm`, neuroevolution, Evolution Strategies. Bài tập: Expected SARSA, sửa reward-to-go, MDP tồn kho, đánh giá off-policy (IPS/SNIPS).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (Q-learning trên FrozenLake trơn, tỉ lệ thắng ≥ 0.74).

## 🔗 Tiếp theo

[36 · Explainable AI](../36_explainable_ai/README.md)

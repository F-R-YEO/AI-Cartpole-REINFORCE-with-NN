# AI-Cartpole-REINFORCE-with-NN
# Vanilla REINFORCE with Neural Network for CartPole-v1

Comparison with DQN, Hyperparametered_DQN, PG, A2C, PPO for evaluation different performance metrics. This includes, average reward score, standard deviation of reward score as measure of variance of model, success rate (>=195), and entropy policy of policy-based model (PG, A2C, PPO, REINFORCE with NN)

This repository contains an implementation of the REINFORCE algorithm using a neural network in TensorFlow/Keras to solve the `CartPole-v1` environment. It includes extensive training and evaluation, along with comparisons against other reinforcement learning models such as DQN, PPO, A2C, and a tuned DQN variant.

## 🚀 Project Highlights

- **Vanilla REINFORCE with Neural Network**: A simple yet effective policy gradient method.
- **Stochastic Policy** with action sampling using predicted probabilities.
- **High Performance**: Achieves >99% success rate and average rewards close to 500.
- **Comprehensive Comparisons**:
  - Baseline DQN
  - Policy Gradient (PG via PPO)
  - Advantage Actor-Critic (A2C)
  - Proximal Policy Optimization (PPO)
  - Hyperparameter-optimized DQN

## 🧩 Environment

- **OpenAI Gym**: `CartPole-v1`
- **Deep Learning Framework**: TensorFlow / Keras
- **Baseline RL Models**: From [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

## 🧠 REINFORCE with Neural Network

A simple MLP-based binary classifier predicts the probability of going left:
```python
modelNNPGR = keras.models.Sequential([
    keras.layers.Dense(16, activation="elu", input_shape=[4]),
    keras.layers.Dense(16, activation="elu"),
    keras.layers.Dense(1, activation="sigmoid")  # probability of action = 0 (left)
])

Training is done using REINFORCE:

- Actions are sampled stochastically.

- Policy gradients are calculated using Monte Carlo returns.

- Gradients are normalized across episodes and applied via Adam optimizer.



Evaluation Metrics
![image](https://github.com/user-attachments/assets/c5543add-7bf3-4bd7-a584-879a22e37b45)


Note: Although REINFORCE is a stochastic policy method, the trained agent became highly confident in its decisions (low entropy) due to convergence in the CartPole task.

📹 Visualizations
Training reward graphs

Entropy comparisons

Policy behavior snapshot

Optional video recordings (RGB array or RecordVideo)

📁 Files
reinforce_nn.ipynb: Full training & evaluation notebook.

vanilla_reinforce_cartpole.keras: Saved trained model.

plot_comparison.py: Script for graphing model performance.

video/: Optional folder for rendered episode recordings.


Conclusion
The vanilla REINFORCE implementation with a neural network proved to be simple yet highly effective, outperforming several policy gradient variants and even A2C in this environment. Its success demonstrates the power of policy gradient methods when paired with a stable architecture and proper reward normalization.

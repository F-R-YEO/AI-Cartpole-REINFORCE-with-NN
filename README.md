# AI-Cartpole-REINFORCE-with-NN
# Vanilla REINFORCE with Neural Network for CartPole-v1
# Comparison with DQN, Hyperparametered_DQN, PG, A2C, PPO for evaluation different performance metrics. This includes, average reward score, standard deviation of reward score as measure of variance of model, success rate (>=195), and entropy policy of policy-based model (PG, A2C, PPO, REINFORCE with NN)

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

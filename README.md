# Pong DQN with PyTorch & Pygame

A reinforcement learning project where an agent learns to play **Pong** using a **Deep Q-Network (DQN)** implemented in **PyTorch**. The environment is built with **Pygame**, allowing you to both train the AI and play against it.

## Features
- Custom Pong environment built with Pygame
- Deep Q-Network (DQN) implementation in PyTorch
- Replay buffer and target network for stable training
- Training and playing modes
- Save and load trained models

## Requirements
Make sure you have **Python 3.8+** installed, then install dependencies:

```bash
pip install pygame torch numpy
```

## Usage

### Training the Agent
Run training for a given number of episodes (e.g., 2000):

```bash
python Pong_DQN_Pygame.py --mode train --episodes 2000
```

This will periodically save model checkpoints (`dqn_pong_epXXXX.pt`) and finally store the trained model as `dqn_pong_final.pt`.

### Playing Against the Trained Agent
Load a trained model and play against it. You control the **left paddle** with `W` (up) and `S` (down):

```bash
python Pong_DQN_Pygame.py --mode play --model dqn_pong_final.pt
```

## Arguments
Some useful arguments:
- `--episodes`: number of training episodes (default: 2000)
- `--batch_size`: minibatch size for DQN training (default: 64)
- `--lr`: learning rate (default: 1e-3)
- `--gamma`: discount factor (default: 0.99)
- `--replay_size`: replay buffer capacity (default: 20000)
- `--eps_start`, `--eps_end`, `--eps_decay`: epsilon-greedy parameters
- `--target_update`: frequency (in episodes) to update target network
- `--save_every`: save model checkpoint every N episodes

## Notes
- Training from scratch may require thousands of episodes before the agent becomes proficient.
- You can tune rewards and hyperparameters to improve performance.

---

Enjoy experimenting with reinforcement learning in Pong! 🏓

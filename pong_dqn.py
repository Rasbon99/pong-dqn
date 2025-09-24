"""
Pong con DQN (PyTorch) e pygame

Istruzioni rapide:
- Requisiti: Python 3.8+, pygame, torch, numpy
  pip install pygame torch numpy

- Modalità:
  1) TRAIN: addestra l'agente (salva modello dopo ogni N episodi)
     python Pong_DQN_Pygame.py --mode train --episodes 5000
  2) PLAY: carica modello e gioca contro il modello (tu sinistra: W/S)
     python Pong_DQN_Pygame.py --mode play --model path_to.pt

Descrizione tecnica (semplificata):
- Ambiente Pong minimal: stato low-dim (ball_x, ball_y, vel_x, vel_y, paddle_y, paddle_vel)
- Azioni: 0 (su), 1 (fermo), 2 (giù)
- Ricompense: +1 se il modello colpisce la palla, -1 se la manca; ricompense di movimento piccole opzionali
- DQN con replay buffer ed epsilon-greedy, target network aggiornato a intervalli

Nota: Questo è un esempio didattico, pensato per essere compatto e facile da modificare.
"""

import argparse
import math
import random
import time
from collections import deque, namedtuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- ENVIRONMENT -----------------------------
class PongEnv:
    """Ambiente Pong minimal con pygame rendering opzionale."""
    def __init__(self, width=640, height=480, paddle_h=80, paddle_w=12, ball_size=12, render=False, fps=60):
        self.width = width
        self.height = height
        self.paddle_w = paddle_w
        self.paddle_h = paddle_h
        self.ball_size = ball_size
        self.render_mode = render
        self.fps = fps

        # positions
        self.reset()

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong DQN")
            self.clock = pygame.time.Clock()

    def reset(self):
        # paddles: player (left) & agent (right)
        self.player_y = self.height // 2 - self.paddle_h // 2
        self.agent_y = self.height // 2 - self.paddle_h // 2
        self.player_v = 0
        self.agent_v = 0

        # ball in centro
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        angle = random.uniform(-0.25 * math.pi, 0.25 * math.pi)
        direction = random.choice([-1, 1])
        speed = 5.0
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

        self.done = False
        self.t = 0

        return self._get_state()

    def _get_state(self):
        # stato continuo normalizzato in [-1,1]
        return np.array([
            (self.ball_x - self.width/2) / (self.width/2),
            (self.ball_y - self.height/2) / (self.height/2),
            self.ball_vx / 10.0,
            self.ball_vy / 10.0,
            (self.agent_y - self.height/2) / (self.height/2),
            self.agent_v / 10.0,
        ], dtype=np.float32)

    def step(self, agent_action, player_action=1):
        """Esegue step. agent_action: 0 su,1 resta,2 giu. player_action same (default 1=fermo).
        Ritorna: next_state, reward, done, info
        """
        # map actions
        def act_to_vel(a):
            if a == 0:
                return -6
            elif a == 2:
                return 6
            else:
                return 0

        self.agent_v = act_to_vel(agent_action)
        self.player_v = act_to_vel(player_action)

        # aggiornamento posizioni
        self.agent_y += self.agent_v
        self.player_y += self.player_v

        # clamp paddles
        self.agent_y = max(0, min(self.height - self.paddle_h, self.agent_y))
        self.player_y = max(0, min(self.height - self.paddle_h, self.player_y))

        # ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # collisione con top/bottom
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1
        if self.ball_y + self.ball_size >= self.height:
            self.ball_y = self.height - self.ball_size
            self.ball_vy *= -1

        reward = 0.0
        done = False

        # collisione con paddle sinistra (player)
        if self.ball_x <= self.paddle_w:
            if self.player_y <= self.ball_y <= self.player_y + self.paddle_h:
                # rimbalzo
                self.ball_x = self.paddle_w
                self.ball_vx *= -1.05
                # qualche variazione verticale in base al punto di impatto
                rel = (self.ball_y - (self.player_y + self.paddle_h/2)) / (self.paddle_h/2)
                self.ball_vy += rel * 2.0
            else:
                # agent wins
                reward = 1.0
                done = True

        # collisione con paddle destra (agent)
        if self.ball_x + self.ball_size >= self.width - self.paddle_w:
            if self.agent_y <= self.ball_y <= self.agent_y + self.paddle_h:
                # rimbalzo
                self.ball_x = self.width - self.paddle_w - self.ball_size
                self.ball_vx *= -1.05
                rel = (self.ball_y - (self.agent_y + self.paddle_h/2)) / (self.paddle_h/2)
                self.ball_vy += rel * 2.0
                reward = 1.0  # reward per hit
            else:
                # agent misses => negative reward
                reward = -1.0
                done = True

        self.t += 1
        # optional: termine per tempo e segnala tie
        if self.t > 1000:
            done = True

        self.done = done

        if self.render_mode:
            self._render()

        return self._get_state(), reward, done, {}

    # ----------------------- Rendering -----------------------
    def _render(self):
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((0, 0, 0))
        # paddles
        pygame.draw.rect(self.screen, (255, 255, 255), (0, int(self.player_y), self.paddle_w, self.paddle_h))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.width - self.paddle_w, int(self.agent_y), self.paddle_w, self.paddle_h))
        # ball
        pygame.draw.ellipse(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y), self.ball_size, self.ball_size))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.render_mode:
            pygame.quit()

# ----------------------------- DQN AGENT -----------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------- TRAIN & PLAY -----------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = PongEnv(render=False)
    state_dim = 6
    n_actions = 3

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(capacity=args.replay_size)

    steps_done = 0
    epsilon = args.eps_start

    def select_action(state):
        nonlocal steps_done, epsilon
        sample = random.random()
        # decay epsilon
        eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * steps_done / args.eps_decay)
        epsilon = eps_threshold
        steps_done += 1
        if sample < eps_threshold:
            return random.randrange(n_actions)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q = policy_net(s)
                return int(q.max(1)[1].item())

    def optimize_model():
        if len(replay) < args.batch_size:
            return
        trans = replay.sample(args.batch_size)
        state_batch = torch.tensor(np.stack(trans.state), dtype=torch.float32, device=device)
        action_batch = torch.tensor(trans.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(trans.reward, dtype=torch.float32, device=device).unsqueeze(1)
        next_state_batch = torch.tensor(np.stack(trans.next_state), dtype=torch.float32, device=device)
        done_batch = torch.tensor(trans.done, dtype=torch.float32, device=device).unsqueeze(1)

        q_values = policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + (1.0 - done_batch) * args.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # training loop
    print("Starting training on device:", device)
    episode_rewards = []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        for t in range(1, args.max_steps_per_episode + 1):
            action = select_action(state)
            # simple heuristic for player (left): follow ball
            player_action = 1
            if env.player_y + env.paddle_h/2 < env.ball_y:
                player_action = 2
            elif env.player_y + env.paddle_h/2 > env.ball_y:
                player_action = 0

            next_state, reward, done, _ = env.step(action, player_action)
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            optimize_model()

            if done:
                break

        episode_rewards.append(total_reward)

        # update target
        if ep % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # save model periodically
        if ep % args.save_every == 0:
            torch.save(policy_net.state_dict(), f"dqn_pong_ep{ep}.pt")
            print(f"Saved model at episode {ep}")

        if ep % args.log_interval == 0:
            avg_r = np.mean(episode_rewards[-args.log_interval:])
            print(f"Ep {ep}/{args.episodes}  avg_reward(last {args.log_interval})={avg_r:.3f}  epsilon={epsilon:.3f}")

    env.close()
    torch.save(policy_net.state_dict(), args.model)
    print("Training completed. Model saved to", args.model)


def play(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = PongEnv(render=True)
    state_dim = 6
    n_actions = 3
    policy_net = DQN(state_dim, n_actions).to(device)
    policy_net.load_state_dict(torch.load(args.model, map_location=device))
    policy_net.eval()

    print("Playing with model", args.model)
    state = env.reset()
    done = False

    while True:
        # handle pygame events and player input
        keys = pygame.key.get_pressed()
        player_action = 1
        if keys[pygame.K_w]:
            player_action = 0
        elif keys[pygame.K_s]:
            player_action = 2

        # agent selects action
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = policy_net(s)
            agent_action = int(q.max(1)[1].item())

        next_state, reward, done, _ = env.step(agent_action, player_action)
        state = next_state

        if done:
            print("Round finished. Resetting...")
            time.sleep(1.0)
            state = env.reset()


# ----------------------------- ARGUMENTS -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'play'], default='train')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max_steps_per_episode', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--replay_size', type=int, default=20000)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=float, default=20000)
    parser.add_argument('--target_update', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--model', type=str, default='dqn_pong_final.pt')
    args = parser.parse_args()

    # attach args to shorter names used in code
    args.batch_size = args.batch_size
    args.episodes = args.episodes
    args.lr = args.lr
    args.gamma = args.gamma
    args.replay_size = args.replay_size
    args.eps_start = args.eps_start
    args.eps_end = args.eps_end
    args.eps_decay = args.eps_decay
    args.target_update = args.target_update
    args.save_every = args.save_every
    args.log_interval = args.log_interval

    if args.mode == 'train':
        train(args)
    else:
        play(args)

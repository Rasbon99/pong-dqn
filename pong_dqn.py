import argparse
import math
import random
import time
import os
from collections import deque, namedtuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- ENVIRONMENT -----------------------------
class PongEnv:
    def __init__(self, width=640, height=480, paddle_h=80, paddle_w=12, ball_size=12, render=False, fps=60):
        self.width = width
        self.height = height
        self.paddle_w = paddle_w
        self.paddle_h = paddle_h
        self.ball_size = ball_size
        self.render_mode = render
        self.fps = fps

        # punteggi
        self.player_score = 0
        self.agent_score = 0
        self.winning_score = 3
        self.winner = None

        # positions
        self.reset()

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            pygame.display.set_caption("Pong DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 32)

            self.game_surface = pygame.Surface((self.width, self.height))

            # Joystick 
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.is_joystick = True
                print("Joystick Connected:", self.joystick.get_name())
            else:
                self.joystick = None
                self.is_joystick = False
                print("No Joystick Found")

            # Sound Effects
            pygame.mixer.init()
            self.sound_hit = pygame.mixer.Sound("sounds/hit.wav")
            self.sound_score = pygame.mixer.Sound("sounds/score.wav")
            self.sound_win = pygame.mixer.Sound("sounds/win.mp3")

    def reset(self, full_reset=False):
        if full_reset:
            self.player_score = 0
            self.agent_score = 0
            self.winner = None

        # paddles
        self.player_y = self.height // 2 - self.paddle_h // 2
        self.agent_y = self.height // 2 - self.paddle_h // 2
        self.player_v = 0
        self.agent_v = 0

        # ball
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
        return np.array([
            (self.ball_x - self.width/2) / (self.width/2),
            (self.ball_y - self.height/2) / (self.height/2),
            self.ball_vx / 10.0,
            self.ball_vy / 10.0,
            (self.agent_y - self.height/2) / (self.height/2),
            self.agent_v / 10.0,
        ], dtype=np.float32)

    def step(self, agent_action, player_action=1):
        def act_to_vel(a):
            if a == 0:
                return -6
            elif a == 2:
                return 6
            else:
                return 0

        self.agent_v = act_to_vel(agent_action)
        self.player_v = act_to_vel(player_action)

        self.agent_y += self.agent_v
        self.player_y += self.player_v
        self.agent_y = max(0, min(self.height - self.paddle_h, self.agent_y))
        self.player_y = max(0, min(self.height - self.paddle_h, self.player_y))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1
        if self.ball_y + self.ball_size >= self.height:
            self.ball_y = self.height - self.ball_size
            self.ball_vy *= -1

        reward = 0.0
        done = False

        # collision on left paddle (player)
        if self.ball_x <= self.paddle_w:
            if self.player_y <= self.ball_y <= self.player_y + self.paddle_h:
                self.ball_x = self.paddle_w
                self.ball_vx *= -1.05
                rel = (self.ball_y - (self.player_y + self.paddle_h/2)) / (self.paddle_h/2)
                self.ball_vy += rel * 2.0
                self.sound_hit.play()
            else:
                # agent segna
                self.agent_score += 1
                done = True
                self.sound_score.play()

        # collision on right paddle (agent)
        if self.ball_x + self.ball_size >= self.width - self.paddle_w:
            if self.agent_y <= self.ball_y <= self.agent_y + self.paddle_h:
                self.ball_x = self.width - self.paddle_w - self.ball_size
                self.ball_vx *= -1.05
                rel = (self.ball_y - (self.agent_y + self.paddle_h/2)) / (self.paddle_h/2)
                self.ball_vy += rel * 2.0
                reward = 1.0
                self.sound_hit.play()
            else:
                # player segna
                self.player_score += 1
                done = True
                self.sound_score.play()

        # win logic
        if self.player_score >= self.winning_score:
            self.winner = "Player"
            done = True
            self.sound_win.play()
        elif self.agent_score >= self.winning_score:
            self.winner = "CPU"
            done = True
            self.sound_win.play()

        self.t += 1
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
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_r:
                    self.reset(full_reset=True)
            elif evt.type == pygame.JOYBUTTONDOWN:
                if evt.button == 9:
                    self.reset(full_reset=True)

        # disegna tutto sulla superficie del gioco
        self.game_surface.fill((0, 0, 0))
        pygame.draw.rect(self.game_surface, (255, 255, 255), (0, int(self.player_y), self.paddle_w, self.paddle_h))
        pygame.draw.rect(self.game_surface, (255, 255, 255), (self.width - self.paddle_w, int(self.agent_y), self.paddle_w, self.paddle_h))
        pygame.draw.ellipse(self.game_surface, (255, 255, 255), (int(self.ball_x), int(self.ball_y), self.ball_size, self.ball_size))

        # score
        score_text = f"{self.player_score} | {self.agent_score}"
        score_surface = self.font.render(score_text, True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(self.width//2, 30))
        self.game_surface.blit(score_surface, score_rect)

        # win message
        if self.winner:
            msg = f"{self.winner} Win!"
            msg_surface = self.font.render(msg, True, (255, 255, 0))
            msg_rect = msg_surface.get_rect(center=(self.width//2, self.height//2))
            self.game_surface.blit(msg_surface, msg_rect)

        # calcola il fattore di scala per mantenere il rapporto 4:3
        window_width, window_height = self.screen.get_size()
        scale_w = window_width / self.width
        scale_h = window_height / self.height
        scale = min(scale_w, scale_h)

        new_w = int(self.width * scale)
        new_h = int(self.height * scale)

        # centra la superficie nella finestra
        x_offset = (window_width - new_w) // 2
        y_offset = (window_height - new_h) // 2

        scaled_surface = pygame.transform.smoothscale(self.game_surface, (new_w, new_h))
        self.screen.fill((0, 0, 0))  # bordo nero
        self.screen.blit(scaled_surface, (x_offset, y_offset))

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
def play(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = PongEnv(render=True)
    state_dim = 6
    n_actions = 3
    policy_net = DQN(state_dim, n_actions).to(device)
    policy_net.load_state_dict(torch.load(args.model, map_location=device))
    policy_net.eval()

    print("Playing with model", args.model)
    state = env.reset(full_reset=True)

    while True:
        # input player
        keys = pygame.key.get_pressed()
        player_action = 1
        if keys[pygame.K_w]:
            player_action = 0
        elif keys[pygame.K_s]:
            player_action = 2

        if env.is_joystick:
            axis_y = env.joystick.get_axis(1)
            if axis_y < -0.5:
                player_action = 0
            elif axis_y > 0.5:
                player_action = 2

        # CPU action (neaural network)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = policy_net(s)
            agent_action = int(q.max(1)[1].item())

        next_state, reward, done, _ = env.step(agent_action, player_action)
        state = next_state

        if done:
            if env.winner:
                print(env.winner, "wins the match!")
                time.sleep(3.0)  # waiting new match
                state = env.reset(full_reset=True)
            else:
                state = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['play'], default='play')
    parser.add_argument('--model', type=str, default='dqn_pong_final.pt')
    args = parser.parse_args()

    play(args)

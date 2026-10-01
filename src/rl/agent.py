# -*- coding: utf-8 -*-
"""DQN 智能体（支持 vanilla / Double / Dueling / Double-Dueling 消融组合）。

- Dueling: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
- Double: 动作由 online 网络选，价值由 target 网络估（否则纯 target）
- 无效动作: target 中屏蔽（mask→-inf），终止转移 target=0
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 4


class QNet(nn.Module):
    def __init__(self, state_dim=16, hidden=256, dueling=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.dueling = dueling
        if dueling:
            self.v = nn.Linear(hidden, 1)
            self.a = nn.Linear(hidden, N_ACTIONS)
        else:
            self.head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, s):
        h = self.body(s)
        if self.dueling:
            v = self.v(h)
            a = self.a(h)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(h)


class Replay:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done, mask2):
        self.buf.append((s, a, r, s2, done, mask2))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, done, m2 = map(np.array, zip(*batch))
        return (torch.from_numpy(s.astype(np.float32)),
                torch.from_numpy(a.astype(np.int64)),
                torch.from_numpy(r.astype(np.float32)),
                torch.from_numpy(s2.astype(np.float32)),
                torch.from_numpy(done.astype(np.float32)),
                torch.from_numpy(m2.astype(np.bool)))


class DQNAgent:
    def __init__(self, state_dim=16, hidden=256, lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=50000,
                 double=True, dueling=True, device="cuda"):
        self.q = QNet(state_dim, hidden, dueling).to(device)
        self.q_target = QNet(state_dim, hidden, dueling).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.double = double
        self.device = device
        self.steps = 0
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps

    @property
    def eps(self):
        f = max(0.0, 1 - self.steps / self.eps_decay_steps)
        return self.eps_end + (self.eps_start - self.eps_end) * f

    def act(self, states, valid_mask, greedy=False):
        """ε-greedy，带无效动作屏蔽。states: (B,dim) numpy, mask: (B,4) bool。"""
        B = len(states)
        if not greedy and random.random() < self.eps:
            acts = np.array([random.choice(np.flatnonzero(m)) for m in valid_mask])
            self.steps += B
            return acts
        with torch.no_grad():
            q = self.q(torch.from_numpy(states).to(self.device))
            q = q.cpu().numpy()
        q[~valid_mask] = -np.inf
        if not greedy:
            self.steps += B
        return q.argmax(1)

    def update(self, batch):
        s, a, r, s2, done, m2 = batch
        s, a, r = s.to(self.device), a.to(self.device), r.to(self.device)
        s2, done, m2 = s2.to(self.device), done.to(self.device), m2.to(self.device)

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                q2_online = self.q(s2)
                q2_online[~m2] = -np.inf
                a2 = q2_online.argmax(1, keepdim=True)
            else:
                a2 = self.q_target(s2).argmax(1, keepdim=True)
                a2[~m2.any(1)] = 0  # 无有效动作时占位（下面会被置 0）
            q2_target = self.q_target(s2).gather(1, a2).squeeze(1)
            q2_target[~m2.any(1)] = 0.0  # 终止转移 target=0
            y = r + self.gamma * (1 - done) * q2_target
        loss = ((q_sa - y) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

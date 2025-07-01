import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.out = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # Buy, Hold, Sell
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        mini_batch = random.sample(self.memory, batch_size)

        states = torch.tensor(np.vstack([s[0] for s in mini_batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([s[1] for s in mini_batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(
            [float(s[2]) if not isinstance(s[2], (int, float)) else s[2] for s in mini_batch],
            dtype=torch.float32
        ).to(self.device)
        next_states = torch.tensor(np.vstack([s[3] for s in mini_batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([float(s[4]) for s in mini_batch]), dtype=torch.float32).to(self.device)

        q_values = self.model(states)
        curr_Q = q_values.gather(1, actions).squeeze(1)

        next_q_values = self.model(next_states)
        max_next_Q = next_q_values.max(1)[0]

        target_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        loss = self.criterion(curr_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

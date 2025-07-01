import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import datetime

# PyTorch Agent Implementation
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_size, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first, then to tensors
        states = np.vstack([e[0] for e in batch])  # Stack into 2D array
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.vstack([e[3] for e in batch])  # Stack into 2D array
        dones = np.array([e[4] for e in batch])
        
        # Debug shapes
        #print(f"States shape: {states.shape}")
        #print(f"Actions shape: {actions.shape}")
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# Helper functions
def getState(data, t, n):
    """Get state representation for time t with window size n"""
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else np.concatenate([np.repeat(data[0], -d), data[0:t + 1]])
    res = []
    for i in range(n - 1):
        diff = block[i + 1] - block[i]
        # Handle numpy arrays properly
        if hasattr(diff, 'item'):
            res.append(diff.item())
        else:
            res.append(float(diff))
    return np.array(res, dtype=np.float32)  # Ensure consistent dtype

def formatPrice(n):
    """Format price for display"""
    # Convert to scalar if it's a numpy array
    if isinstance(n, np.ndarray):
        n = n.item()
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# Mode selection for data slicing
mode = "relative"  # options: "date" or "relative"
target_date = "2025-04-04"  # Used if mode == "date"

target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
end_date = target_date + datetime.timedelta(days=1)

# Settings
ticker = "WIPRO.NS"
interval = "5m"
period = "7d"
skip = 4

# Fetch data
print(f"Downloading {ticker} data...")

if mode == "date":
    data_df = yf.download(tickers=ticker, interval=interval, start=target_date, end=end_date)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if target_date not in unique_dates:
        raise ValueError(f"Target date {target_date} not found in data.")
    train_data = data_df[data_df['Date'] == target_date]['Close'].dropna().values
    test_data = None  # No separate test data in this mode

elif mode == "relative":
    data_df = yf.download(tickers=ticker, interval=interval, period=period)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough days of data to split into train and test.")
    train_data = data_df[data_df['Date'].isin(unique_dates[:-skip])]['Close'].dropna().values
    test_data = data_df[data_df['Date'].isin(unique_dates[-skip:])]['Close'].dropna().values
else:
    raise ValueError("Invalid mode selected. Choose either 'date' or 'relative'.")

print(f"Training samples: {len(train_data)}")

# Training settings
window_size = 7
episode_count = 25
batch_size = 32

# Initialize agent
agent = Agent(window_size)
l = len(train_data) - 1

# Precompute states
states = [getState(train_data, t, window_size + 1) for t in range(window_size, len(train_data))]

# Debug: Check state shape
print(f"State shape: {states[0].shape}")
print(f"Expected input size: {window_size}")

# Create models folder
os.makedirs("models", exist_ok=True)

# Tracking
reward_history = []
profit_history = []

print(len(train_data), "training samples")

starting_balance = 2000

for e in range(episode_count + 1):
    print(f"Episode {e}/{episode_count}")
    balance = starting_balance
    state = states[window_size]

    total_profit = 0
    total_reward = 0
    agent.inventory = []

    for t in range(window_size, l):
        action = agent.act(state)  # state is now 1D array
        next_state = states[t + 1 - window_size]
        reward = 0

        if action == 1 and balance >= train_data[t]:  # Buy if balance allows
            price = train_data[t].item() if hasattr(train_data[t], 'item') else float(train_data[t])
            agent.inventory.append(price)
            balance -= price
            print(f"Buy: {formatPrice(price)} | New Balance: {formatPrice(balance)}")

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            price = train_data[t].item() if hasattr(train_data[t], 'item') else float(train_data[t])
            bought_price = agent.inventory.pop(0)
            reward = max(price - bought_price, -1)  # Penalize bad trades
            total_profit += price - bought_price
            balance += price
            print(f"Sell: {formatPrice(price)} | Profit: {formatPrice(price - bought_price)} | New Balance: {formatPrice(balance)}")

        total_reward += reward
        done = True if t == l - 1 else False
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f"Ending Balance after Episode {e}: {formatPrice(balance)}")
            print("--------------------------------")
            print(f"Total Profit: {formatPrice(total_profit)}")
            print(f"Total Reward: {formatPrice(total_reward)}")
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    reward_history.append(total_reward)
    profit_history.append(total_profit)

    # Save model every episode
    agent.save_model(f"models/model_ep{e}.pth")

# Plot training profit and reward history
plt.figure(figsize=(12, 6))
plt.plot(profit_history, label="Total Profit per Episode", color="green")
plt.plot(reward_history, label="Total Reward per Episode", color="blue")
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.title("Training Progress: Profit and Reward per Episode")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()
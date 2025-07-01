import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from agent.agent_torch import Agent
from functions import getState, formatPrice
import datetime
import torch
from collections import Counter

# Mode selection for data slicing
mode = "relative"  # options: "date" or "relative"
target_date = "2025-04-04"  # Used if mode == "date"

target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
end_date = target_date  + datetime.timedelta(days=1)

# Settings
ticker = "WIPRO.NS"
interval = "5m"
period = "14d"
skip = 10

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
window_size = 20
episode_count = 25
batch_size = 32

# Initialize agent
agent = Agent(window_size)

# Ensemble training settings
base_model_number = 26 # Choose model number to refine
base_model_path = f"models/model_ep{base_model_number}.pt"

l = len(train_data) - 1

# Precompute states
states = [getState(train_data, t, window_size + 1) for t in range(window_size, len(train_data))]

# Create models folder
os.makedirs("models", exist_ok=True)

# Tracking
reward_history = []
profit_history = []

print(len(train_data), "training samples")

starting_balance = 2000

base_model_number = 26  # Choose model number to refine
base_model_path = f"models/model_ep{base_model_number}.pt"

for e in range(episode_count + 1):
    if e == 0 and os.path.exists(base_model_path):
        print(f"Loading base model from: {base_model_path}")
        agent.model = torch.load(base_model_path)
        boosting_weights = np.ones(len(states))
        predictions_log = []
    elif e == 0:
        print(f"Base model not found at {base_model_path}. Training from scratch.")
    print(f"Episode {e}/{episode_count}")
    balance = starting_balance
    state = states[window_size]

    total_profit = 0
    total_reward = 0
    agent.inventory = []
    action_counter = Counter()

    for t in range(window_size, l):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = agent.model(state_tensor)
        action_probs = action_probs.squeeze()
        print(f"Action Probabilities: Buy={action_probs[1].item():.4f}, Sell={action_probs[2].item():.4f}, Hold={action_probs[0].item():.4f}")
        action = torch.argmax(action_probs).item()
        action_counter[action] += 1

        predictions_log.append((t, action))

        # Boosting weight update: penalize wrong action
        correct_action = None
        if action == 1 and balance >= train_data[t]:
            correct_action = 1
        elif action == 2 and len(agent.inventory) > 0:
            correct_action = 2
        elif action == 0:
            correct_action = 0

        if action != correct_action:
            boosting_weights[t - window_size] *= 1.2  # Increase weight for wrong prediction
        else:
            boosting_weights[t - window_size] *= 0.9  # Decrease weight for correct prediction

        next_state = states[t + 1 - window_size]
        reward = 0

        if action == 1 and balance >= train_data[t]:  # Buy if balance allows
            agent.inventory.append(train_data[t])
            balance -= train_data[t]
            reward += 1  # No immediate reward for buying
            print(f"Buy: {formatPrice(train_data[t])} | New Balance: {formatPrice(balance)}")

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = agent.inventory.pop(0)
            reward = train_data[t] - bought_price  # Actual profit or loss
            if reward < 0:
                reward *= 2  # Amplify penalty slightly for losses
            total_profit += train_data[t] - bought_price
            balance += train_data[t]
            print(f"Sell: {formatPrice(train_data[t])} | Profit: {formatPrice(train_data[t] - bought_price)} | New Balance: {formatPrice(balance)}")

        elif action == 0:  # Hold
            price_change = abs(train_data[t] - train_data[t-1])
            if price_change < 0.1:
                reward -= 10  # Small penalty for idle holding during low volatility
            else:
                reward -= 1   # Minor cost of inaction

        total_reward += reward
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            # Ensure equal number of buys and sells
            if len(agent.inventory) > 0:
                print(f"WARNING: Unequal number of buys and sells. Inventory remaining: {len(agent.inventory)}")
            # Enforce parity: discard any unmatched buys
            if len(agent.inventory) > 0:
                unmatched_buys = len(agent.inventory)
                print(f"WARNING: {unmatched_buys} unmatched buys discarded to enforce parity.")
                agent.inventory = []  # Discard any unmatched buys

            print(f"Ending Balance after Episode {e}: {formatPrice(balance)}")
            print("--------------------------------")
            print(f"Total Profit: {formatPrice(total_profit)}")
            print(f"Total Reward: {formatPrice(total_reward)}")
            print(f"Action distribution: {action_counter}")
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    reward_history.append(total_reward)
    profit_history.append(total_profit)

    # Save model every episode
    torch.save(agent.model, f"models/model_ep{e}.pt")

# Sanity check: filter out any invalid or nested entries
profit_history = [float(p) for p in profit_history if isinstance(p, (int, float, np.number))]
reward_history = [float(r) for r in reward_history if isinstance(r, (int, float, np.number))]

boosting_weights /= np.sum(boosting_weights)
print("\nBoosting Weights Summary:")
print("Max:", np.max(boosting_weights))
print("Min:", np.min(boosting_weights))
print("Mean:", np.mean(boosting_weights))

# Plot training profit and reward history
plt.figure(figsize=(15, 8))

# Subplot for Profit
plt.subplot(2, 1, 1)
plt.plot(profit_history, label="Total Profit per Episode", color="green")
plt.xlabel("Episodes")
plt.ylabel("Profit")
plt.title("Training Progress: Profit per Episode")
plt.grid()
plt.legend()

# Subplot for Reward
plt.subplot(2, 1, 2)
plt.plot(reward_history, label="Total Reward per Episode", color="blue")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Progress: Reward per Episode")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()

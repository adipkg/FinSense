import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from agent.agent import Agent
from functions import getState, formatPrice
import datetime

# Mode selection for data slicing
mode = "relative"  # options: "date" or "relative"
target_date = "2025-04-04"  # Used if mode == "date"

target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
end_date = target_date  + datetime.timedelta(days=1)

# Settings
ticker = "TATASTEEL.NS"
interval = "5m"
period = "7d"
skip = 1

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
episode_count = 5
batch_size = 32

# Initialize agent
agent = Agent(window_size)
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

for e in range(episode_count + 1):
    print(f"Episode {e}/{episode_count}")
    balance = starting_balance
    state = states[window_size]

    total_profit = 0
    total_reward = 0
    agent.inventory = []

    for t in range(window_size, l):
        action = agent.act(state)
        next_state = states[t + 1 - window_size]
        reward = 0

        if action == 1 and balance >= train_data[t]:  # Buy if balance allows
            agent.inventory.append(train_data[t])
            balance -= train_data[t]
            print(f"Buy: {formatPrice(train_data[t])} | New Balance: {formatPrice(balance)}")

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = agent.inventory.pop(0)
            reward = max(train_data[t] - bought_price, -1)  # Penalize bad trades
            total_profit += train_data[t] - bought_price
            balance += train_data[t]
            print(f"Sell: {formatPrice(train_data[t])} | Profit: {formatPrice(train_data[t] - bought_price)} | New Balance: {formatPrice(balance)}")

        total_reward += reward
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            # Add penalty for leftover inventory
            #if len(agent.inventory) > 0:
             #   penalty = -10 * len(agent.inventory)  # Penalize for each unsold stock
              #  total_reward += penalty
               # print(f"Penalty for {len(agent.inventory)} unsold stocks: {penalty}")

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
    agent.model.save(f"models/model_ep{e}.keras")

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
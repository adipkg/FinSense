import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from agent.agent import Agent
from functions import getStockDataVec, getState, formatPrice

n = 6

if len(sys.argv) != 2:
    print("Usage: python evaluate.py [stock]")
    exit()

# Load stock data
stock_name = sys.argv[1]
data = getStockDataVec(stock_name)
#data = data[101:300]
data = data[::-1] 
l = len(data) - 1

# Set batch size
batch_size = 32
window_size = 10  # Adjust based on training settings

# Directory containing models
model_dir = "models/"
profit_per_episode = []  # Store total profit per episode

# Loop through models from episode 0 to 20
for episode in range(1,n):
    model_path = f"{model_dir}model_ep{episode}.keras"
    model_name = f"model_ep{episode}.keras"

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found, skipping...")
        profit_per_episode.append(None)
        continue

    print(f"\nEvaluating Model: model_ep{episode}.keras")

    # Load model
    model = load_model(model_path)
    window_size = model.layers[0].input.shape[1]

    # Initialize agent for evaluation
    agent = Agent(window_size, is_eval=True, model_name=model_name)

    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    # Tracking buy/sell points for plotting
    buy_dates = []
    sell_dates = []
    all_prices = []

    # Evaluation loop
    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # Buy action
            agent.inventory.append(data[t])
            buy_dates.append(t)
            print(f"Buy: {formatPrice(data[t])}")

        elif action == 2 and len(agent.inventory) > 0:  # Sell action
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            sell_dates.append(t)
            print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

        state = next_state
        all_prices.append(data[t])

    # Store profit for this episode
    profit_per_episode.append(total_profit)
    print("--------------------------------")
    print(f"Episode {episode} - {stock_name} Total Profit: {formatPrice(total_profit)}")
    print("--------------------------------")

# Plot total profit per episode
plt.figure(figsize=(10, 5))
plt.plot(range(1,n), profit_per_episode, marker="o", linestyle="-", color="blue", label="Total Profit per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Profit")
plt.title("Profit per Episode (Model Performance Over Time)")
plt.xticks(range(1,n))  # Ensure all episodes (0 to 20) are displayed on the x-axis
plt.legend()
plt.grid()
plt.savefig("profit_per_episode.png")
plt.show()
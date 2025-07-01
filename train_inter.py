import os
import matplotlib.pyplot as plt
from agent.agent import Agent
from functions import *
import sys

# Ensure correct usage
if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

# Read command-line arguments
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# Initialize agent
agent = Agent(window_size)
data = getStockDataVec(stock_name)
#data = data[:100][::-1] 
data = data[::-1] 
l = len(data) - 1
batch_size = 32

# Ensure "models" directory exists
os.makedirs("models", exist_ok=True)

# Track training progress
reward_history = []
profit_history = []

for e in range(episode_count + 1):
    print(f"Episode {e}/{episode_count}")
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    total_reward = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # Buy
            agent.inventory.append(data[t])
            print(f"Buy: {formatPrice(data[t])}")

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, -1)  # Penalize bad trades
            total_profit += data[t] - bought_price
            print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

        total_reward += reward
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print(f"Total Profit: {formatPrice(total_profit)}")
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    # Store episode metrics for visualization
    reward_history.append(total_reward)
    profit_history.append(total_profit)

    # Save model every episode
    agent.model.save(f"models/model_ep{e}.keras")

# Plot training metrics
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(reward_history, label="Total Reward per Episode", color="blue")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(profit_history, label="Total Profit per Episode", color="green")
plt.xlabel("Episodes")
plt.ylabel("Total Profit")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()
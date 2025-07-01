import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from agent.agent import Agent
from functions import getStockDataVec, getState, formatPrice

# Ensure correct usage
if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

# Load stock data and model
stock_name, model_name = sys.argv[1], sys.argv[2]
model_path = "models/" + model_name

model = load_model(model_path)
window_size = model.layers[0].input.shape[1]

# Initialize agent for evaluation
agent = Agent(window_size, is_eval=True, model_name=model_name)
data = getStockDataVec(stock_name)
#data = data[101:150]
data = data[::-1] 

l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

# Tracking buy/sell points for plotting
buy_prices = []
sell_prices = []
buy_dates = []
sell_dates = []
all_prices = []

# Counters for buy and sell actions
buy_count = 0
sell_count = 0

# Evaluation loop
for t in range(l):
    action = agent.act(state)
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    if action == 1:  # Buy action
        agent.inventory.append(data[t])
        buy_prices.append(data[t])
        sell_prices.append(None)  # Maintain alignment
        buy_dates.append(t)
        buy_count += 1
        print(f"Buy: {formatPrice(data[t])}")

    elif action == 2 and len(agent.inventory) > 0:  # Sell action
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        buy_prices.append(None)  # Maintain alignment
        sell_prices.append(data[t])
        sell_dates.append(t)
        sell_count += 1
        print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

    else:  # Hold action
        buy_prices.append(None)
        sell_prices.append(None)

    state = next_state
    all_prices.append(data[t])

# Print final profit and trade counts
print("--------------------------------")
print(f"{stock_name} Total Profit: {formatPrice(total_profit)}")
print(f"Total Buy Actions: {buy_count}")
print(f"Total Sell Actions: {sell_count}")
print("--------------------------------")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(all_prices, label="Stock Price", color="blue", alpha=0.6)
plt.scatter(buy_dates, [all_prices[i] for i in buy_dates], color="green", marker="^", label="Buy", s=100)
plt.scatter(sell_dates, [all_prices[i] for i in sell_dates], color="red", marker="v", label="Sell", s=100)

# Update the title to include total profit and number of buy/sell actions
plt.title(f"Trading Actions: {stock_name}\nTotal Profit: {formatPrice(total_profit)} | Buys: {buy_count} | Sells: {sell_count}")
plt.xlabel("Time (Days)")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()
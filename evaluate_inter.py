import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from agent.agent import Agent
from functions import getStockDataVec, getState, formatPrice

os.makedirs("visuals", exist_ok=True)

# Ensure correct usage
if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

# Load stock data and model
stock_name, model_name = sys.argv[1], sys.argv[2]
model_path = "models/" + model_name

model = load_model(model_path)
window_size = model.input_shape[1]

# Initialize agent
agent = Agent(window_size, is_eval=True, model_name=model_name)
agent.inventory = []
agent.balance = 50000  # Not used here, but safe to set

data = getStockDataVec(stock_name)
#data = data[101:500]
data = data[::-1]
l = len(data) - 1
state = getState(data, 0, window_size + 1)

# Tracking
total_profit = 0
buy_prices, sell_prices = [], []
buy_dates, sell_dates = [], []
all_prices = []
portfolio_values = []
profit_over_time = []
buy_count = 0
sell_count = 0

# ✅ Buy-and-Hold graph tracking (retained)
buy_and_hold_portfolio_values = []
initial_price = data[0]
shares_bought = 1
for price in data:
    value = shares_bought * price
    buy_and_hold_portfolio_values.append(value)

# ✅ Buy-and-Hold profit for value comparison (final - initial)
buy_and_hold_profit = data[-1] - data[0]

# Evaluation loop
for t in range(l):
    action = agent.act(state)
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    if action == 1:
        agent.inventory.append(data[t])
        buy_prices.append(data[t])
        sell_prices.append(None)
        buy_dates.append(t)
        buy_count += 1
        print(f"Buy: {formatPrice(data[t])}")
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        buy_prices.append(None)
        sell_prices.append(data[t])
        sell_dates.append(t)
        sell_count += 1
        print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")
    else:
        buy_prices.append(None)
        sell_prices.append(None)

    state = next_state
    all_prices.append(data[t])

    inventory_value = sum(agent.inventory) if agent.inventory else 0
    portfolio_values.append(inventory_value)

    profit_over_time.append(total_profit)

# Final value comparison
final_model_value = profit_over_time[-1]

np.savez("visuals/evaluation_data.npz",
         all_prices=all_prices,
         buy_dates=buy_dates,
         sell_dates=sell_dates,
         profit_over_time=profit_over_time,
         portfolio_values=portfolio_values,
         buy_and_hold_portfolio_values=buy_and_hold_portfolio_values)

# Summary
print("--------------------------------")
print(f"Total Trading Profit: {formatPrice(total_profit)}")
print(f"Total Buy Actions: {buy_count}")
print(f"Total Sell Actions: {sell_count}")
print("--------------------------------")
print(f"Buy-and-Hold Profit (final - initial): {formatPrice(buy_and_hold_profit)}")
print(f"Model Final Profit (Buy & Hold + Trading): {formatPrice(final_model_value)}")
print(f"Difference: {formatPrice(final_model_value - buy_and_hold_profit)}")
print("--------------------------------")

adjusted_buy_dates = [i - window_size for i in buy_dates]
adjusted_sell_dates = [i - window_size for i in sell_dates]

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(30, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

# Big Trading Actions plot on the left
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(all_prices, label="Stock Price", color="blue", alpha=0.6)
ax1.scatter(adjusted_buy_dates, [all_prices[i] for i in adjusted_buy_dates], color="green", marker="^", label="Buy", s=100)
ax1.scatter(adjusted_sell_dates, [all_prices[i] for i in adjusted_sell_dates], color="red", marker="v", label="Sell", s=100)
ax1.set_title(f"Trading Actions\nTotal Profit: {formatPrice(total_profit)} | Buys: {buy_count} | Sells: {sell_count}")
ax1.set_xlabel("Time (Steps)")
ax1.set_ylabel("Stock Price")
ax1.legend()
ax1.grid()

# Top right plot - Portfolio Value Over Time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(len(portfolio_values)), portfolio_values, label="Model Portfolio Value", color="purple")
ax2.set_title("Portfolio Value Over Time")
ax2.set_xlabel("Time (Steps)")
ax2.set_ylabel("Portfolio Value")
ax2.legend()
ax2.grid()

# Top right plot - Model vs Buy-and-Hold
ax3 = fig.add_subplot(gs[0, 2])
combined_portfolio = [m + b for m, b in zip(profit_over_time, buy_and_hold_portfolio_values)]
ax3.plot(range(len(combined_portfolio)), combined_portfolio, label="Model + Buy & Hold", color="purple")
ax3.plot(range(len(buy_and_hold_portfolio_values)), buy_and_hold_portfolio_values, label="Buy & Hold", color="gray", linestyle="--")
ax3.set_title("Model + Profit vs Buy-and-Hold")
ax3.set_xlabel("Time (Steps)")
ax3.set_ylabel("Portfolio Value")
ax3.legend()
ax3.grid()

# Bottom right plot - Profit Over Time
ax4 = fig.add_subplot(gs[1, 1:])
ax4.plot(range(len(profit_over_time)), profit_over_time, label="Cumulative Profit", color="green")
ax4.set_title("Profit Over Time")
ax4.set_xlabel("Time (Steps)")
ax4.set_ylabel("Total Profit")
ax4.legend()
ax4.grid()

plt.tight_layout()
plt.savefig("visuals/combined_dashboard.png")
plt.show()
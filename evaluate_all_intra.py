import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from agent.agent import Agent
from functions import getStockDataVec, getState
import yfinance as yf

# Fetch today's stock data
ticker = "TATASTEEL.NS"
data_df = yf.download(tickers=ticker, period="1d", interval="1m")
data = data_df['Close'].dropna().values
l = len(data)

# Directory containing models
model_dir = "models"
profits = []
model_numbers = []

for file in os.listdir(model_dir):
    if file.startswith("model_ep") and file.endswith(".keras"):
        model_number = int(file.split("model_ep")[1].split(".keras")[0])
        model_path = os.path.join(model_dir, file)

        model = load_model(model_path)
        window_size = model.input_shape[1]
        agent = Agent(window_size, is_eval=True, model_name=file)
        agent.inventory = []
        agent.balance = 2000

        total_profit = 0

        for t in range(window_size, l - 1):
            state = getState(data, t, window_size + 1)
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)

            if action == 1 and agent.balance >= data[t]:
                agent.inventory.append(data[t])
                agent.balance -= data[t]
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                profit = data[t] - bought_price
                total_profit += profit
                agent.balance += data[t]

        model_numbers.append(model_number)
        profits.append(float(total_profit))
        print(f"Model {model_number}: Profit = {float(total_profit):.2f}")

# Sort results by model number
model_numbers, profits = zip(*sorted(zip(model_numbers, profits)))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(model_numbers, profits, marker='o')
plt.title(f"Model Performance on {ticker}")
plt.xlabel("Model Number")
plt.ylabel("Total Profit")
plt.grid(True)
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/profit_per_model.png")
plt.show()
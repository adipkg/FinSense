import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn

from agent.agent_torch import Agent
from functions import formatPrice, getState
from plotter import plot_evaluation_results

# Constants
MODEL_NUMBER = 1
TICKER = "WIPRO.NS"
START_BALANCE = 2000
INTERVAL = "5m"
VISUALS_DIR = "visuals"


# Mode selection for data slicing
MODE = "today"  # options: "date", "relative", "yesterday", "today"
TARGET_DATE = "2025-04-04"  # Used if MODE == "date"

TARGET_DATE = datetime.datetime.strptime(TARGET_DATE, "%Y-%m-%d").date()
END_DATE = TARGET_DATE + datetime.timedelta(days=1)

PERIOD = "2d"
SKIP = 1

# Model and Agent Setup
Path(VISUALS_DIR).mkdir(exist_ok=True)

model_name = f"model_ep{MODEL_NUMBER}.pt"
model_path = f"models/{model_name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

window_size = model.fc1.in_features  # Assuming first layer input size is window_size

agent = Agent(window_size, is_eval=True, model_name=model_name)
agent.model = model 
agent.inventory = []
agent.balance = START_BALANCE

# Data Loading
print(f"Downloading {TICKER} data...")

if MODE == "date":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, start=TARGET_DATE, end=END_DATE)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if TARGET_DATE not in unique_dates:
        raise ValueError(f"Target date {TARGET_DATE} not found in data.")
    data = data_df[data_df['Date'] == TARGET_DATE]['Close'].dropna().values

elif MODE == "relative":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, period=PERIOD)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough days of data to evaluate.")
    data = data_df[data_df['Date'].isin(unique_dates[:-SKIP])]['Close'].dropna().values

elif MODE == "yesterday":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, period="2d")
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough data to evaluate yesterday.")
    data = data_df[data_df['Date'] == unique_dates[-2]]['Close'].dropna().values

elif MODE == "today":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, period="1d")
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 1:
        raise ValueError("No data to evaluate today.")
    data = data_df[data_df['Date'] == unique_dates[-1]]['Close'].dropna().values

else:
    raise ValueError("Invalid MODE selected. Choose 'date', 'relative', 'yesterday' or 'today'.")

# Evaluation Loop
def evaluate_agent(agent, data, window_size):
    l = len(data) - 1
    state = getState(data, window_size, window_size + 1)

    total_profit = 0
    buy_prices, sell_prices = [], []
    buy_dates, sell_dates = [], []
    all_prices = []
    portfolio_values = []
    profit_over_time = []
    buy_count = 0
    sell_count = 0

    buy_and_hold_portfolio_values = []
    shares_bought = 1
    for price in data:
        value = shares_bought * price
        buy_and_hold_portfolio_values.append(value)
    buy_and_hold_profit = data[-1] - data[0]

    for t in range(window_size, l):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = agent.model(state_tensor)
        action = torch.argmax(q_values).item()

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            if agent.balance >= data[t]:
                agent.inventory.append(data[t])
                agent.balance -= data[t]
            buy_prices.append(data[t])
            sell_prices.append(None)
            buy_dates.append(t)
            buy_count += 1
            print(f"Buy: {formatPrice(data[t])}")
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            agent.balance += data[t]
            buy_prices.append(None)
            sell_prices.append(data[t])
            sell_dates.append(t)
            sell_count += 1
            print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")
        else:
            buy_prices.append(None)
            sell_prices.append(None)

        state = next_state
        all_prices.append(float(data[t]))

        inventory_value = sum(agent.inventory) if agent.inventory else 0
        portfolio_values.append(float(inventory_value))

        profit_over_time.append(float(total_profit))

    inventory_value = sum(agent.inventory) if agent.inventory else 0
    portfolio_values.append(float(inventory_value))
    profit_over_time.append(float(total_profit))

    final_model_value = profit_over_time[-1]

    return {
        "total_profit": total_profit,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "all_prices": all_prices,
        "portfolio_values": portfolio_values,
        "profit_over_time": profit_over_time,
        "buy_and_hold_portfolio_values": buy_and_hold_portfolio_values,
        "buy_and_hold_profit": buy_and_hold_profit,
        "final_model_value": final_model_value,
    }

results = evaluate_agent(agent, data, window_size)

# Summary
print("--------------------------------")
print(f"{TICKER} Total Trading Profit: {formatPrice(results['total_profit'])}")
print(f"Total Buy Actions: {results['buy_count']}")
print(f"Total Sell Actions: {results['sell_count']}")
print("--------------------------------")
print(f"Buy-and-Hold Profit (final - initial): {formatPrice(results['buy_and_hold_profit'])}")
print(f"Model Final Profit (Buy & Hold + Trading): {formatPrice(results['final_model_value'])}")
print(f"Difference: {formatPrice(results['final_model_value'] - results['buy_and_hold_profit'])}")
print("--------------------------------")

# Plot Results
plot_evaluation_results(results, window_size, TICKER)
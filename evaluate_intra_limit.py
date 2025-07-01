import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras.models import load_model

from agent.agent import Agent
from functions import formatPrice, getState
from plotter import plot_evaluation_results



# Constants
MODEL_NUMBER = 1
TICKER = "TATASTEEL.NS"
START_BALANCE = 2000
INTERVAL = "5m"
VISUALS_DIR = "visuals"



# Model and Agent Setup
Path(VISUALS_DIR).mkdir(exist_ok=True)

model_name = f"model_ep{MODEL_NUMBER}.keras"
model_path = f"models/model_ep{MODEL_NUMBER}.keras"

model = load_model(model_path)
window_size = model.input_shape[1]

agent = Agent(window_size, is_eval=True, model_name=model_name)
agent.inventory = []
agent.balance = START_BALANCE



# Data Loading
def load_stock_data(ticker, start, end, interval):
    data_df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval
    )
    return data_df['Close'].dropna().values


today = datetime.date.today()
yesterday = today - datetime.timedelta(days=3)
data = load_stock_data(TICKER, yesterday, today, INTERVAL)



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

    # Buy-and-Hold tracking
    buy_and_hold_portfolio_values = []
    shares_bought = 1
    for price in data:
        value = shares_bought * price
        buy_and_hold_portfolio_values.append(value)
    buy_and_hold_profit = data[-1] - data[0]

    for t in range(window_size, l):
        action = agent.act(state)
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
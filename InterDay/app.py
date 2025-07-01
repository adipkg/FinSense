import streamlit as st
import os
import yfinance as yf
import numpy as np
from keras.models import load_model
from agent.agent import Agent
from functions import getState, formatPrice
import matplotlib.pyplot as plt


# --- Recommendation Logic ---
def get_latest_data(symbol="^GSPC", days=11):
    df = yf.download(symbol, period="15d", interval="1d")
    close_prices = df["Close"].values[-days:]
    return list(close_prices)

def recommend_action(model_name, window_size):
    model_path = f"models/{model_name}"
    model = load_model(model_path)
    agent = Agent(window_size, is_eval=True, model_name=model_name)
    agent.model = model

    data = get_latest_data(days=window_size + 1)
    if len(data) < window_size + 1:
        return "Not enough data"

    state = getState(data, 0, window_size + 1)
    action = agent.act(state)

    if action == 0:
        return "HOLD"
    elif action == 1:
        return "BUY"
    elif action == 2:
        return "SELL"
    else:
        return "UNKNOWN"

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title("FinSense Dashboard")

# Layout split
left_col, right_col = st.columns([1, 2])

# --- Left: Recommendation Section ---
with left_col:
    st.header("Today's Recommendation")
    model_name = "model_ep1.keras"
    window_size = 5
    recommendation = recommend_action(model_name, window_size)
    st.subheader(f"Action: {recommendation}")

    st.markdown("---")
    st.subheader("Backtest Over Last N Days")

    n_days = st.slider("Select number of past days to simulate", 0, 252, 30)
    if n_days >= window_size + 1:
        df = yf.download("^GSPC", period="1y", interval="1d")
        close_prices = df["Close"].values[-(n_days + window_size + 1):]
        data = list(close_prices)

        model_path = f"models/{model_name}"
        model = load_model(model_path)
        agent = Agent(window_size, is_eval=True, model_name=model_name)
        agent.model = model

        state = getState(data, 0, window_size + 1)

        l = len(data) - 1
        
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
                print(f"Buy: {formatPrice(float(data[t]))}")
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                buy_prices.append(None)
                sell_prices.append(data[t])
                sell_dates.append(t)
                sell_count += 1
                print(f"Sell: {formatPrice(float(data[t]))} | Profit: {formatPrice(float(data[t] - bought_price))}")
            else:
                buy_prices.append(None)
                sell_prices.append(None)

            state = next_state
            all_prices.append(data[t])

            inventory_value = sum(agent.inventory) if agent.inventory else 0
            portfolio_values.append(inventory_value)

            profit_over_time.append(total_profit)
            # Ensure float values
            profit_over_time = [float(p) for p in profit_over_time]
            portfolio_values = [float(v) for v in portfolio_values]

        # Final value comparison
        final_model_value = profit_over_time[-1]

        # Summary
        print("--------------------------------")
        print(f"Total Trading Profit: {formatPrice(float(total_profit))}")
        print(f"Total Buy Actions: {buy_count}")
        print(f"Total Sell Actions: {sell_count}")
        print("--------------------------------")
        print(f"Buy-and-Hold Profit (final - initial): {formatPrice(float(buy_and_hold_profit))}")
        print(f"Model Final Profit (Buy & Hold + Trading): {formatPrice(float(final_model_value))}")
        print(f"Difference: {formatPrice(float(final_model_value - buy_and_hold_profit))}")
        print("--------------------------------")

        st.markdown("#### Backtest Summary")
        st.metric("Total Trading Profit", formatPrice(float(total_profit)))
        st.metric("Total Buy Actions", buy_count)
        st.metric("Total Sell Actions", sell_count)
        st.metric("Buy-and-Hold Profit", formatPrice(float(buy_and_hold_profit)))
        st.metric("Model Final Profit", formatPrice(float(final_model_value)))
        st.metric("Difference", formatPrice(float(final_model_value - buy_and_hold_profit)))

        # 1. Trading Actions
        plt.figure(figsize=(12, 6), facecolor="#0e1117")
        ax = plt.gca()
        ax.set_facecolor("#0e1117")
        plt.plot(all_prices, label="Stock Price", color="deepskyblue", alpha=0.6)
        plt.scatter(buy_dates, [all_prices[i] for i in buy_dates], color="lightgreen", marker="^", label="Buy", s=100)
        plt.scatter(sell_dates, [all_prices[i] for i in sell_dates], color="lightcoral", marker="v", label="Sell", s=100)
        plt.title(f"Trading Actions: Total Profit: {formatPrice(float(total_profit))} | Buys: {buy_count} | Sells: {sell_count}", color="lightgray")
        plt.xlabel("Time (Days)", color="lightgray")
        plt.ylabel("Stock Price", color="lightgray")
        plt.tick_params(axis='x', labelcolor='lightgray')
        plt.tick_params(axis='y', labelcolor='lightgray')
        plt.legend(facecolor="#0e1117", edgecolor="lightgray", labelcolor='lightgray')
        plt.grid()
        plt.tight_layout()
        plt.savefig("visuals/trading_actions.png")
        # plt.show()

        # 2. Portfolio Value Over Time
        plt.figure(figsize=(12, 4), facecolor="#0e1117")
        ax = plt.gca()
        ax.set_facecolor("#0e1117")
        try:
            plt.plot(portfolio_values, label="Model Portfolio Value", color="deepskyblue")
        except Exception:
            plt.plot([0], [0], label="No Portfolio Data", color="lightgray")
        plt.title("Portfolio Value Over Time", color="lightgray")
        plt.xlabel("Time (Days)", color="lightgray")
        plt.ylabel("Portfolio Value", color="lightgray")
        plt.tick_params(axis='x', labelcolor='lightgray')
        plt.tick_params(axis='y', labelcolor='lightgray')
        plt.grid()
        plt.legend(facecolor="#0e1117", edgecolor="lightgray", labelcolor='lightgray')
        plt.tight_layout()
        plt.savefig("visuals/portfolio_value.png")
        # plt.show()

        # 3. Model + Buy-and-Hold vs Buy-and-Hold
        combined_portfolio = [m + b for m, b in zip(profit_over_time, buy_and_hold_portfolio_values)]
        plt.figure(figsize=(12, 4), facecolor="#0e1117")
        ax = plt.gca()
        ax.set_facecolor("#0e1117")
        plt.plot(combined_portfolio, label="Model + Buy & Hold", color="deepskyblue")
        plt.plot(buy_and_hold_portfolio_values, label="Buy & Hold", color="lightgray", linestyle="--")
        plt.title("Model + Profit vs Buy-and-Hold", color="lightgray")
        plt.xlabel("Time (Days)", color="lightgray")
        plt.ylabel("Portfolio Value", color="lightgray")
        plt.tick_params(axis='x', labelcolor='lightgray')
        plt.tick_params(axis='y', labelcolor='lightgray')
        plt.legend(facecolor="#0e1117", edgecolor="lightgray", labelcolor='lightgray')
        plt.grid()
        plt.tight_layout()
        plt.savefig("visuals/combined_vs_bh.png")
        # plt.show()

        # 4. Profit Over Time
        plt.figure(figsize=(12, 4), facecolor="#0e1117")
        ax = plt.gca()
        ax.set_facecolor("#0e1117")
        try:
            plt.plot(profit_over_time, label="Cumulative Profit", color="lightgreen")
        except Exception:
            plt.plot([0], [0], label="No Profit Data", color="lightgray")
        plt.title("Profit Over Time", color="lightgray")
        plt.xlabel("Time (Days)", color="lightgray")
        plt.ylabel("Total Profit", color="lightgray")
        plt.tick_params(axis='x', labelcolor='lightgray')
        plt.tick_params(axis='y', labelcolor='lightgray')
        plt.grid()
        plt.legend(facecolor="#0e1117", edgecolor="lightgray", labelcolor='lightgray')
        plt.tight_layout()
        plt.savefig("visuals/profit_over_time.png")
        # plt.show()
    else:
        st.info("Please select at least 6 days to simulate.")

# --- Right: Performance Based on Backtest ---
with right_col:
    st.header("Performance Based on Backtest")
    if n_days >= window_size + 1:
        image_files = [
            "trading_actions.png",
            "portfolio_value.png",
            "combined_vs_bh.png",
            "profit_over_time.png"
        ]
        for img in image_files:
            path = os.path.join("visuals", img)
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.warning(f"Plot not found: {path}")
    else:
        st.info("Not enough days selected to show performance.")
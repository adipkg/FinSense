import yfinance as yf
import numpy as np
from keras.models import load_model
from agent.agent import Agent
from functions import getState

def get_latest_data(symbol="^GSPC", days=11):
    df = yf.download(symbol, period="15d", interval="1d")
    close_prices = df["Close"].values[-days:]  # get last `days` closing prices
    return list(close_prices)

def recommend_action(model_name, window_size):
    model_path = f"models/{model_name}"
    model = load_model(model_path)
    agent = Agent(window_size, is_eval=True, model_name=model_name)
    agent.model = model

    # Get latest data
    data = get_latest_data(days=window_size + 1)
    if len(data) < window_size + 1:
        print("Not enough data to make a prediction.")
        return

    state = getState(data, 0, window_size + 1)
    action = agent.act(state)

    if action == 0:
        print("🟡 Recommendation: HOLD the S&P 500.")
    elif action == 1:
        print("🟢 Recommendation: BUY the S&P 500.")
    elif action == 2:
        print("🔴 Recommendation: SELL the S&P 500.")

if __name__ == "__main__":
    model_name = "model_ep1.keras" # change to your latest trained model
    window_size = 5  # must match the window size you trained with
    recommend_action(model_name, window_size)
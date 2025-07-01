import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the same classes from training code
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Helper functions
def getState(data, t, n):
    """Get state representation for time t with window size n"""
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else np.concatenate([np.repeat(data[0], -d), data[0:t + 1]])
    res = []
    for i in range(n - 1):
        diff = block[i + 1] - block[i]
        # Handle numpy arrays properly
        if hasattr(diff, 'item'):
            res.append(diff.item())
        else:
            res.append(float(diff))
    return np.array(res, dtype=np.float32)

def formatPrice(n):
    """Format price for display"""
    # Convert to scalar if it's a numpy array
    if isinstance(n, np.ndarray):
        n = n.item()
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def plot_evaluation_results(results, window_size, ticker):
    """Plot evaluation results"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Main price chart with buy/sell signals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results['all_prices'], label='Price', color='blue', alpha=0.7)
    
    # Mark buy and sell points
    for i, date in enumerate(results['buy_dates']):
        if date < len(results['all_prices']):
            ax1.scatter(date, results['all_prices'][date], color='green', marker='^', s=100, zorder=5)
    
    for i, date in enumerate(results['sell_dates']):
        if date < len(results['all_prices']):
            ax1.scatter(date, results['all_prices'][date], color='red', marker='v', s=100, zorder=5)
    
    ax1.set_title(f'{ticker} Price with Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Profit comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results['profit_over_time'], label='Trading Profit', color='green')
    ax2.axhline(y=results['buy_and_hold_profit'], color='orange', linestyle='--', label='Buy & Hold Profit')
    ax2.set_title('Profit Comparison')
    ax2.set_ylabel('Profit ($)')
    ax2.set_xlabel('Time Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Portfolio value
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(results['portfolio_values'], label='Portfolio Value', color='purple')
    ax3.plot(results['buy_and_hold_portfolio_values'][window_size:len(results['portfolio_values'])+window_size], 
             label='Buy & Hold Value', color='orange', linestyle='--')
    ax3.set_title('Portfolio Value')
    ax3.set_ylabel('Value ($)')
    ax3.set_xlabel('Time Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'evaluation_results_{ticker}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Constants
MODEL_NUMBER = 25  # Use the last episode model
TICKER = "WIPRO.NS"
START_BALANCE = 2000
INTERVAL = "5m"
VISUALS_DIR = "visuals"

# Mode selection for data slicing
MODE = "date"  # options: "date", "relative", "yesterday", "today"
TARGET_DATE = "2025-06-20"  # Used if MODE == "date"

TARGET_DATE = datetime.datetime.strptime(TARGET_DATE, "%Y-%m-%d").date()
END_DATE = TARGET_DATE + datetime.timedelta(days=1)

PERIOD = "2d"
SKIP = 1

# Model and Agent Setup
Path(VISUALS_DIR).mkdir(exist_ok=True)

model_name = f"model_ep{MODEL_NUMBER}.pth"  # Changed to .pth extension
model_path = f"models/{model_name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Create model with correct architecture (window_size=7, action_size=3)
window_size = 7
action_size = 3
model = DQNNetwork(window_size, action_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from {model_path}")

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

print(f"Loaded {len(data)} data points for evaluation")

# Evaluation Loop
def evaluate_agent(model, data, window_size, start_balance):
    l = len(data) - 1
    state = getState(data, window_size, window_size + 1)
    
    balance = start_balance
    inventory = []
    total_profit = 0
    buy_prices, sell_prices = [], []
    buy_dates, sell_dates = [], []
    all_prices = []
    portfolio_values = []
    profit_over_time = []
    buy_count = 0
    sell_count = 0

    # Buy and hold comparison
    buy_and_hold_portfolio_values = []
    shares_bought = start_balance / data[0] if data[0] > 0 else 1
    for price in data:
        value = shares_bought * price
        buy_and_hold_portfolio_values.append(value)
    buy_and_hold_profit = (data[-1] - data[0]) * shares_bought

    for t in range(window_size, l):
        # Get action from model
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state = getState(data, t + 1, window_size + 1)
        price = data[t].item() if hasattr(data[t], 'item') else float(data[t])

        if action == 1 and balance >= price:  # Buy
            inventory.append(price)
            balance -= price
            buy_prices.append(price)
            sell_prices.append(None)
            buy_dates.append(t)
            buy_count += 1
            print(f"Buy: {formatPrice(price)} | Balance: {formatPrice(balance)}")
        elif action == 2 and len(inventory) > 0:  # Sell
            bought_price = inventory.pop(0)
            profit = price - bought_price
            total_profit += profit
            balance += price
            buy_prices.append(None)
            sell_prices.append(price)
            sell_dates.append(t)
            sell_count += 1
            print(f"Sell: {formatPrice(price)} | Profit: {formatPrice(profit)} | Balance: {formatPrice(balance)}")
        else:  # Hold
            buy_prices.append(None)
            sell_prices.append(None)

        state = next_state
        all_prices.append(price)

        # Calculate portfolio value (cash + inventory value)
        inventory_value = sum(inventory) if inventory else 0
        portfolio_value = balance + inventory_value
        portfolio_values.append(portfolio_value)
        profit_over_time.append(total_profit)

    # Final calculations
    final_inventory_value = sum(inventory) if inventory else 0
    final_portfolio_value = balance + final_inventory_value
    final_profit = final_portfolio_value - start_balance

    return {
        "total_profit": total_profit,
        "final_profit": final_profit,
        "final_portfolio_value": final_portfolio_value,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "all_prices": all_prices,
        "portfolio_values": portfolio_values,
        "profit_over_time": profit_over_time,
        "buy_and_hold_portfolio_values": buy_and_hold_portfolio_values,
        "buy_and_hold_profit": buy_and_hold_profit,
    }

# Run evaluation
results = evaluate_agent(model, data, window_size, START_BALANCE)

# Summary
print("================================")
print(f"{TICKER} Evaluation Results")
print("================================")
print(f"Total Trading Profit: {formatPrice(results['total_profit'])}")
print(f"Final Portfolio Value: {formatPrice(results['final_portfolio_value'])}")
print(f"Final Profit (vs initial): {formatPrice(results['final_profit'])}")
print(f"Total Buy Actions: {results['buy_count']}")
print(f"Total Sell Actions: {results['sell_count']}")
print("--------------------------------")
print(f"Buy-and-Hold Profit: {formatPrice(results['buy_and_hold_profit'])}")
print(f"Trading vs Buy-and-Hold: {formatPrice(results['final_profit'] - results['buy_and_hold_profit'])}")
print("================================")

# Plot Results
plot_evaluation_results(results, window_size, TICKER)
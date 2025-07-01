import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from functions import formatPrice


def plot_evaluation_results(results, window_size, ticker):
    buy_dates = results['buy_dates']
    sell_dates = results['sell_dates']
    all_prices = results['all_prices']
    portfolio_values = results['portfolio_values']
    profit_over_time = results['profit_over_time']
    buy_and_hold_portfolio_values = results['buy_and_hold_portfolio_values']
    total_profit = results['total_profit']
    buy_count = results['buy_count']
    sell_count = results['sell_count']

    adjusted_buy_dates = [i - window_size for i in buy_dates]
    adjusted_sell_dates = [i - window_size for i in sell_dates]

    fig = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

    # Big Trading Actions plot on the left
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(all_prices, label="Stock Price", color="blue", alpha=0.6)
    ax1.scatter(adjusted_buy_dates, [all_prices[i] for i in adjusted_buy_dates], color="green", marker="^", label="Buy", s=100)
    ax1.scatter(adjusted_sell_dates, [all_prices[i] for i in adjusted_sell_dates], color="red", marker="v", label="Sell", s=100)
    
    ax1.set_title(f"Trading Actions\nTotal Profit: {formatPrice(total_profit)} | Buys: {int(buy_count)} | Sells: {int(sell_count)}")
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
    plt.savefig(f"visuals/{ticker}_combined_dashboard.png")
    plt.show()
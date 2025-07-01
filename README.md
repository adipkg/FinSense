# Deep Q Network Based Stock Trader

Financial markets are complex, dynamic systems influenced by non-linear patterns, economic factors, and unpredictable events. Traditional financial models, such as regression-based or time-series approaches, often struggle to capture the evolving nature of market behavior, particularly during volatile periods. 

This project presents a Deep Q Network (DQN)-based reinforcement learning framework for stock trading, supporting both interday (daily) and intraday (minute-level) strategies using historical and live market data. Rather than predicting prices directly, the model learns trading policies—buy, sell, or hold—through interaction with the market environment, optimizing long-term cumulative rewards.

By integrating reinforcement learning with deep neural networks, the agent develops adaptive strategies that respond to market shifts in real time. The framework emphasizes modularity and scalability, making it lightweight enough for practical deployment across various financial assets. This approach moves beyond traditional prediction-based trading towards policy-based decision-making, enabling the agent to adjust dynamically to changing conditions. The project aims to contribute to intelligent portfolio management by demonstrating the effectiveness of reinforcement learning in developing robust, data-driven trading strategies.


---

## Project Structure
- `agent.py`: Contains the Q-learning agent with neural network architecture, action selection policy, and training logic.
- `functions.py`: Includes helper functions for data preprocessing, state construction, and reward computation.
- `train.py`: Trains models on daily interday stock data.
- `evaluate.py`: Evaluates interday models on a provided dataset.
- `train_inter.py`: Interday-specific training script.
- `evaluate_inter.py`: Interday-specific evaluation script.
- `train_intra.py`: Trains models on 5-minute intraday stock data from Yahoo Finance.
- `evaluate_intra.py`: Evaluates intraday models on selected date.
- `evaluate_all_inter.py`: Batch evaluation of all interday models.
- `evaluate_all_intra.py`: Batch evaluation of all intraday models.
- `evaluate_intra_limit.py`: Intraday evaluation with portfolio constraints or trade limits.

---

## Features
- Deep Q-learning with experience replay and epsilon-greedy action selection
- Model checkpointing per training episode
- Visualizations for trading decisions, cumulative profit, and portfolio value
- Compatibility with all Yahoo Finance-listed stocks
- Dual-mode support: interday (long-term) and intraday (real-time) trading

---

## Usage

### Interday Mode

Train and evaluate models using daily historical data (CSV format).

```bash
# Train model
python train.py [stock.csv] [window_size] [episodes]

# Example
python train.py ^GSPC.csv 10 50

# Evaluate trained model
python evaluate.py [stock.csv]

# Example
python evaluate.py ^GSPC.csv


```

### Intraday Mode

Train and evaluate models on recent 5-minute stock price data using live Yahoo Finance API.

```bash
# Train on the last 30 days of intraday 5-minute data
python train_intra.py

# Evaluate on a specific date (configurable inside the script)
python evaluate_intra.py

```

## Visual Output

During evaluation, the following graphs are generated and saved:
- Trading Actions: Buy/Sell actions plotted on stock price.
- Portfolio Value Evolution: Change in the agent’s portfolio value over time.
- Cumulative Profit: Accumulated profit over the trading period.
- Comparison with Buy-and-Hold: Model performance benchmarked against a passive strategy.

Visualizations are saved automatically inside the visuals directory.


## Dependencies
	•	Python 3.8+
	•	TensorFlow
	•	NumPy
	•	Matplotlib
	•	Pandas
	•	yfinance



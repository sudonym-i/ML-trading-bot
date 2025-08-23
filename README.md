
# ML Trading Bot

This project is an automated trading bot that uses deep learning to predict future stock prices and make trading decisions. The bot is designed for experimentation and educational purposes.

## How it Works

- **Data Loading:**
    - Downloads historical price data for tickers specified in `config.json` using Yahoo Finance (`yfinance`).
    - Adds technical indicators (SMA, RSI, MACD) to the data for richer feature representation.
    - Data is split into sequences of configurable length for time series prediction.

- **Machine Learning Model:**
    - The core model is a multi-layer Gated Recurrent Unit (GRU) neural network implemented in PyTorch (`model.py`).
        - 3 GRU layers, 128 hidden units each
        - Fully connected layers with ReLU activation and dropout for regularization
        - Trained to predict the next closing price given a sequence of past prices and indicators

- **Training:**
    - Model is trained using Mean Squared Error (MSE) loss and Adam optimizer.
    - Training parameters (epochs, batch size, learning rate, etc.) are set in `config.json`.

- **Trading Logic:**
    - For each time step, the model predicts the next price.
    - **Buy** if prediction > current price (or hold if already holding)
    - **Sell** if prediction < current price
    - The bot currently trades the entire portfolio (all-in/all-out, no position sizing)

## Configuration

Edit `src/config.json` to set:

- `tickers`: List of stock symbols
- `start`, `end`: Date range for historical data
- `seq_length`: Number of time steps in each input sequence
- `interval`: Data frequency (e.g., '1d')
- `epochs`, `batch_size`, `lr`: Training hyperparameters

## Limitations

- The trading strategy is intentionally simple and does not account for transaction costs, slippage, or risk management.
- The model does not use advanced portfolio optimization or reinforcement learning (yet).

---
For more details, see the code in `src/`.
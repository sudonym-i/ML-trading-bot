# TSR Model (Time Series Regression Model)

A comprehensive time series regression model for stock price prediction and automated trading using GRU (Gated Recurrent Unit) neural networks.

## Overview

The TSR model is a deep learning-based system that:
1. Fetches stock market data using Yahoo Finance API
2. Calculates technical indicators (SMA, RSI, MACD)
3. Trains a GRU neural network to predict future stock prices
4. Simulates automated trading strategies based on predictions
5. Provides comprehensive visualizations and performance metrics

## Architecture

### Model Structure (`model.py`)
- **GRUPredictor**: A PyTorch neural network with the following architecture:
  - **Input Layer**: Variable input dimensions based on features (Close, SMA_14, RSI_14, MACD)
  - **GRU Layer**: 3-layer GRU with 128 hidden units and batch_first=True
  - **Fully Connected Layers**: 
    - FC1: 128 ’ 64 neurons with SiLU activation
    - Dropout: 20% for regularization
    - FC2: 64 ’ 1 neuron (price prediction output)
  - **Forward Pass**: Takes last time step output from GRU sequence

### Data Processing Pipeline

#### Data Handler (`data_handler.py`)
- **DataLoader Class**: Downloads stock data from Yahoo Finance
  - Supports multiple tickers, date ranges, and intervals (1d, 1h, 5m)
  - Handles data cleaning and NaN removal
  
- **make_dataset Function**: Core data preprocessing pipeline
  - Downloads OHLCV data for specified tickers
  - Adds technical indicators using `utils.add_technical_indicators()`
  - Creates sequences of specified length (default: 24 time steps)
  - Supports normalization (z-score) for training stability
  - Combines multiple tickers into single dataset
  - Returns PyTorch TensorDataset and input dimensions

#### Technical Indicators (`utils.py`)
- **Simple Moving Average (SMA)**: 14-period rolling average
- **Relative Strength Index (RSI)**: 14-period momentum oscillator (0-100 scale)
- **MACD**: Moving Average Convergence Divergence (12-day EMA - 26-day EMA)
- **Sequence Creation**: Converts DataFrame to sliding window sequences for time series prediction

### Training Pipeline (`train.py`)

The `train_gru_predictor` function implements:
- **Optimizer**: Adam with configurable learning rate (default: 1e-3)
- **Loss Function**: Mean Squared Error (MSE)
- **Training Loop**: 
  - Batch processing with configurable batch size
  - Gradient computation and backpropagation
  - Loss tracking per epoch
  - Optional training loss visualization

### Trading Simulation (`trade.py`)

#### TradingSimulator Class
- **Portfolio Management**: 
  - Tracks balance, position (shares held), and trade history
  - Supports buy/sell/hold decisions based on model predictions
  
- **Trading Logic**:
  - **Buy Signal**: Predicted price > current price AND no current position
  - **Sell Signal**: Predicted price < current price AND holding position
  - **Position Sizing**: Uses entire balance for purchases (all-in strategy)

#### simulate_trading Function
- **Model Evaluation**: Sets model to eval mode for inference
- **Data Normalization**: Applies window-wise z-score normalization per sequence
- **Prediction Denormalization**: Converts normalized predictions back to original price scale
- **Performance Tracking**: Records portfolio values, trades, and profits
- **Visualization**: Generates performance charts if requested

### Visualization Suite (`visualizations.py`)

#### Training Visualizations
- **plot_training_loss**: Displays loss reduction over epochs with automatic save

#### Trading Performance
- **plot_portfolio_performance**: Dual-panel chart showing:
  - Portfolio value over time with buy/sell markers
  - Profit/loss per completed trade
  
- **plot_price_predictions**: Comparison of actual vs predicted prices
- **plot_performance_metrics**: Comprehensive analysis including:
  - Profit/loss distribution histogram
  - Cumulative profit curve
  - Performance summary table (win rate, profit factor, etc.)
  - Maximum win/loss streak analysis

#### Technical Analysis
- **plot_technical_indicators**: Multi-panel technical analysis chart:
  - Price and SMA overlay
  - RSI with overbought/oversold levels (70/30)
  - MACD with zero line reference

#### Interactive Dashboard
- **create_interactive_dashboard**: Plotly-based interactive visualization combining all metrics

### Configuration System

#### Training Configuration (`train_nn.json`)
```json
{
  "tickers": ["AAPL", "MSFT", "NVDA"],
  "start": "2022-01-01", 
  "end": "2023-01-01",
  "seq_length": 24,
  "interval": "1d",
  "epochs": 40,
  "batch_size": 7,
  "lr": 0.0001
}
```

#### Testing Configuration (`test_nn.json`)  
```json
{
  "tickers": ["MSFT"],
  "start": "2022-01-01",
  "end": "2023-01-01", 
  "seq_length": 24,
  "interval": "1d",
  "epochs": 4,
  "batch_size": 1,
  "lr": 0.0001
}
```

### Main Interface (`route.py`)

#### train_model()
1. Loads training configuration from `train_nn.json`
2. Creates normalized dataset with technical indicators
3. Initializes GRUPredictor with correct input dimensions
4. Trains model with specified hyperparameters
5. Returns trained model

#### test_model(model)
1. Loads testing configuration from `test_nn.json`
2. Creates unnormalized test dataset
3. Runs trading simulation with trained model
4. Generates performance visualizations

## Data Flow

1. **Configuration Loading**: JSON configs specify tickers, dates, and hyperparameters
2. **Data Acquisition**: Yahoo Finance API downloads OHLCV data
3. **Feature Engineering**: Technical indicators added to raw price data
4. **Sequence Generation**: Time series converted to supervised learning format
5. **Normalization**: Z-score normalization applied for training stability
6. **Model Training**: GRU learns to predict next-period price from sequence
7. **Trading Simulation**: Model predictions drive buy/sell decisions
8. **Performance Analysis**: Comprehensive metrics and visualizations generated

## Key Features

- **Multi-Ticker Support**: Train on multiple stocks simultaneously
- **Configurable Architecture**: Adjustable sequence length, batch size, learning rate
- **Technical Analysis Integration**: SMA, RSI, and MACD indicators
- **Robust Normalization**: Window-wise z-score normalization for stable training
- **Comprehensive Trading Simulation**: Realistic portfolio tracking with transaction costs
- **Rich Visualizations**: Training metrics, trading performance, and technical analysis
- **Interactive Dashboards**: Plotly-based interactive analysis tools

## Dependencies

See `requirements.txt`:
- **torch**: PyTorch deep learning framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance data API
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive visualizations

## Usage

```python
# Train a model
trained_model = train_model()

# Test the model with trading simulation  
test_model(trained_model)
```

The system automatically handles data download, preprocessing, training, and generates comprehensive performance reports with visualizations.
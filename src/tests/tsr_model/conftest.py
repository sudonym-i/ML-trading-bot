"""
Pytest configuration and shared fixtures for TSR model tests.

This file contains shared test fixtures and configuration that can be used
across all TSR model test modules.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

# Add src directory to Python path for imports
# This resolves to /home/sudonym/repos/ML-trading-bot/src
src_path = str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "tsr_model")
print(f"DEBUG: Adding to sys.path: {src_path}")
print(f"DEBUG: Path exists: {Path(src_path).exists()}")
print(f"DEBUG: tsr_model exists: {(Path(src_path) / 'tsr_model').exists()}")
sys.path.insert(0, src_path)

from tsr_model.model import GRUPredictor
from tsr_model.data_pipeline import DataLoader
from tsr_model.utils import add_technical_indicators, create_sequences


@pytest.fixture
def sample_stock_data():
    """
    Create sample stock data for testing.
    
    Returns:
        pd.DataFrame: Sample stock data with OHLCV columns
    """
    np.random.seed(42)  # For reproducible tests
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create realistic-looking stock data
    base_price = 100
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% drift, 2% volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_stock_data_with_indicators(sample_stock_data):
    """
    Sample stock data with technical indicators added.
    
    Returns:
        pd.DataFrame: Stock data with SMA, RSI, MACD indicators
    """
    return add_technical_indicators(sample_stock_data.copy())


@pytest.fixture
def sample_sequences(sample_stock_data_with_indicators):
    """
    Create sample sequence data for model training/testing.
    
    Returns:
        tuple: (X, y) sequences for model input/output
    """
    seq_length = 10
    return create_sequences(sample_stock_data_with_indicators, seq_length)


@pytest.fixture
def sample_gru_model():
    """
    Create a sample GRU model for testing.
    
    Returns:
        GRUPredictor: Initialized GRU model
    """
    input_dim = 8  # OHLCV + 3 indicators (SMA, RSI, MACD)
    return GRUPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)


@pytest.fixture
def mock_yfinance_data():
    """
    Mock data that simulates yfinance download response.
    
    Returns:
        pd.DataFrame: Mock yfinance data
    """
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    
    return pd.DataFrame({
        'Open': np.random.uniform(95, 105, 50),
        'High': np.random.uniform(100, 110, 50),
        'Low': np.random.uniform(90, 100, 50),
        'Close': np.random.uniform(95, 105, 50),
        'Volume': np.random.randint(1000000, 3000000, 50)
    }, index=dates)


@pytest.fixture
def test_config():
    """
    Test configuration dictionary.
    
    Returns:
        dict: Configuration for testing
    """
    return {
        "tickers": ["AAPL"],
        "start": "2023-01-01",
        "end": "2023-12-31", 
        "seq_length": 10,
        "interval": "1d",
        "epochs": 5,
        "batch_size": 16,
        "lr": 0.001
    }


@pytest.fixture(autouse=True)
def setup_torch():
    """
    Setup PyTorch for deterministic behavior in tests.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
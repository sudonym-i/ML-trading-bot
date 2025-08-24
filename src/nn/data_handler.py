
import yfinance as yf
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from .model import GRUPredictor
from .utils import create_sequences, add_technical_indicators
import numpy as np


class DataLoader:
    def __init__(self, tickers, start, end, interval="1d"):
        """
        tickers: str or list of str
        start: start date (YYYY-MM-DD)
        end: end date (YYYY-MM-DD)
        interval: data frequency (e.g., '1d', '1h', '5m')
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start = start
        self.end = end
        self.interval = interval
        self.data = {}

    def download(self):
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start, end=self.end, interval=self.interval)
            if df is not None and not df.empty:
                df = df.dropna()
                self.data[ticker] = df
        return self.data

    def get(self, ticker):
        return self.data.get(ticker)

    def get_all(self):
        return self.data




def make_dataset(ticker, start, end, seq_length=24, interval="1d", normalize=False):  # e.g., 24 for 24 hours or days
	
    # Accepts a list of tickers, downloads and processes each, and combines all samples
	if isinstance(ticker, str):
		tickers = [ticker]
	else:
		tickers = ticker

	all_X, all_y = [], []
      
	for t in tickers:
		print(f"[INFO] Downloading data for {t} from {start} to {end} with interval {interval}...")
		dl = DataLoader(t, start, end, interval=interval)
		data = dl.download()[t]
		print(f"[INFO] Data shape: {data.shape}")
		
        # Add technical indicators
		data = add_technical_indicators(data)
		print(f"[INFO] Data shape after indicators: {data.shape}")
		print(f"[INFO] Creating sequences with sequence length {seq_length}...")
		features = data[['Close', 'SMA_14', 'RSI_14', 'MACD']]

		if(normalize == True):
			# Normalize features (z-score)
			features_norm = (features - features.mean()) / (features.std() + 1e-8)
			X, y = create_sequences(features_norm, seq_length)
			# Normalize target (z-score)
			y_norm = (y - y.mean()) / (y.std() + 1e-8)
		else:
            # Keeping variables the same for sanity
			X, y = create_sequences(features, seq_length)
			y_norm = y

		print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
            
		all_X.append(X)
		all_y.append(y_norm)
	X = np.concatenate(all_X, axis=0)
	y = np.concatenate(all_y, axis=0)
	X = torch.tensor(X, dtype=torch.float32)
	y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
      
	if y.ndim == 3:
		y = y.squeeze(1)
            
	dataset = TensorDataset(X, y)
	print(f"[INFO] Combined dataset created with {len(dataset)} samples from {len(tickers)} tickers.")
	return dataset, X.shape[2]  # input_dim

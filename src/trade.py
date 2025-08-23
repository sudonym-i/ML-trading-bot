import numpy as np
import torch

class TradingSimulator:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Number of shares held
        self.trades = []

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []

    def step(self, price, prediction):
        """
        price: current stock price
        prediction: model's predicted next price (float)
        """
        # Ensure price and prediction are floats for formatting
        price = float(np.asarray(price).squeeze())
        prediction = float(np.asarray(prediction).squeeze())
        print(f" Current price: ${price:.2f}, Predicted price: ${prediction:.2f}")
        
        action = None
        trade_info = {'action': None, 'price': price, 'balance': self.balance, 'position': self.position, 'profit': 0, 'buy_price': None, 'sell_price': None}
        
        # Buy if prediction is higher than current price and not already holding
        if prediction > price and self.position == 0:
            self.position = abs(self.balance / price)
            self.buy_price = price
            self.balance = 0
            action = 'buy'
            print(f"[TRADE] Bought stock at ${price:.2f} per share, shares: {self.position:.4f}")
            trade_info['buy_price'] = price
        
        # Sell if prediction is lower than current price and holding
        elif prediction < price and self.position > 0:
            sell_value = self.position * price
            profit = sell_value - (self.position * self.buy_price)
            self.balance = sell_value
            self.position = 0
            action = 'sell'
            print(f"[TRADE] Sold stock for ${price:.2f} per share, total: ${sell_value:.2f}, balance-change: ${profit:.2f}")
            trade_info['sell_price'] = price
            trade_info['profit'] = profit
        
        else:
            action = 'hold'
        trade_info['action'] = action
        trade_info['balance'] = self.balance
        trade_info['position'] = self.position
        self.trades.append(trade_info)
        return action
    
    
    
    def print_summary(self, current_price=None):
        final_value = self.get_portfolio_value(current_price) if current_price is not None else self.balance
        total_profit = final_value - self.initial_balance
        print(f"[SUMMARY] Final portfolio value: ${final_value:.2f}")
        print(f"[SUMMARY] Total profit/loss: ${total_profit:.2f}")
        num_trades = sum(1 for t in self.trades if t['action'] in ['buy', 'sell'])
        print(f"[SUMMARY] Number of trades: {num_trades}")
        if self.trades:
            last_trade = self.trades[-1]
            print(f"[SUMMARY] Last action: {last_trade['action']} at price ${last_trade['price']:.2f}")

    def get_portfolio_value(self, current_price):
        return self.balance + self.position * current_price

    def get_trades(self):
        return self.trades



def simulate_trading(model, dataset, prices):
    """
    Simulate trading using the trained model and price data.
    model: trained RNN model
    dataset: TensorDataset of (X, y)
    prices: array-like of actual prices (for portfolio value calculation)
    """
    model.eval()
    sim = TradingSimulator()
    portfolio_values = []
    seq_length = dataset.tensors[0].shape[1]  # sequence length from X shape
    prices = np.array(prices)
    
    with torch.no_grad():
        for i, (X, _) in enumerate(dataset):
            X = X.unsqueeze(0)
            z_pred = float(model(X).item())
            # Use the 'Close' price from the last time step in the sequence (feature 0)
            # For rolling mean/std, get the window of actual prices ending at this step
            price_idx = i + seq_length - 1
            if price_idx >= len(prices):
                break  # avoid index error if prices is shorter than dataset
            window_start = price_idx - seq_length + 1
            window_end = price_idx + 1
            price_window = prices[window_start:window_end]
            rolling_mean = price_window.mean()
            rolling_std = price_window.std() + 1e-8
            pred_price = z_pred * rolling_std + rolling_mean
            price = prices[price_idx]
            sim.step(price, pred_price)
            portfolio_values.append(sim.get_portfolio_value(price))

    print(f"[INFO] Final portfolio value: ${float(portfolio_values[-1]):.2f}")
    return portfolio_values, sim.get_trades()

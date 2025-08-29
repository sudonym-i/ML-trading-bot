import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import threading
from collections import deque


class RealTimeVisualizer:
    """Real-time visualization for trading bot performance and market data"""
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.price_data = deque(maxlen=max_points)
        self.portfolio_data = deque(maxlen=max_points)
        self.prediction_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.trades = []
        
    def update_data(self, timestamp, price, portfolio_value, prediction=None):
        """Update the visualization data"""
        self.timestamps.append(timestamp)
        self.price_data.append(price)
        self.portfolio_data.append(portfolio_value)
        if prediction is not None:
            self.prediction_data.append(prediction)
    
    def add_trade(self, timestamp, price, action, profit=None):
        """Add a trade event"""
        self.trades.append({
            'timestamp': timestamp,
            'price': price,
            'action': action,
            'profit': profit or 0
        })
    
    def create_live_dashboard(self, ticker):
        """Create a live dashboard figure"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'{ticker} - Live Price & Predictions',
                'Portfolio Value',
                'Recent Trades'
            ),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Convert deques to lists for plotting
        timestamps = list(self.timestamps)
        prices = list(self.price_data)
        portfolio = list(self.portfolio_data)
        predictions = list(self.prediction_data)
        
        # Price and predictions
        if timestamps and prices:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, 
                    y=prices, 
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            if predictions and len(predictions) == len(timestamps):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps, 
                        y=predictions, 
                        mode='lines',
                        name='Predicted Price',
                        line=dict(color='red', dash='dash', width=2)
                    ),
                    row=1, col=1
                )
        
        # Portfolio value
        if timestamps and portfolio:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, 
                    y=portfolio, 
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
        
        # Trade markers
        if self.trades:
            recent_trades = self.trades[-50:]  # Show last 50 trades
            buy_trades = [t for t in recent_trades if t['action'] == 'buy']
            sell_trades = [t for t in recent_trades if t['action'] == 'sell']
            
            if buy_trades:
                buy_times = [t['timestamp'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=10, color='green')
                    ),
                    row=1, col=1
                )
            
            if sell_trades:
                sell_times = [t['timestamp'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=10, color='red')
                    ),
                    row=1, col=1
                )
            
            # Profit/Loss bar chart
            profit_data = [t['profit'] for t in recent_trades if t['profit'] != 0]
            if profit_data:
                colors = ['green' if p > 0 else 'red' for p in profit_data]
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(profit_data))),
                        y=profit_data,
                        name='Trade P&L',
                        marker_color=colors
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Live Trading Dashboard',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio ($)", row=2, col=1)
        fig.update_yaxes(title_text="Profit/Loss ($)", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return fig
    
    def save_dashboard(self, ticker, filename=None):
        """Save the current dashboard as HTML"""
        if filename is None:
            filename = f"{ticker}_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        fig = self.create_live_dashboard(ticker)
        fig.write_html(filename)
        print(f"Dashboard saved as {filename}")
        return filename


class LiveDataFeed:
    """Simulate live data feed for testing"""
    
    def __init__(self, ticker, interval='1m'):
        self.ticker = ticker
        self.interval = interval
        self.is_running = False
        self.callback = None
        
    def set_callback(self, callback):
        """Set callback function to receive live data"""
        self.callback = callback
    
    def start_feed(self, duration_minutes=60):
        """Start the live data feed"""
        self.is_running = True
        thread = threading.Thread(target=self._feed_loop, args=(duration_minutes,))
        thread.daemon = True
        thread.start()
        return thread
    
    def stop_feed(self):
        """Stop the live data feed"""
        self.is_running = False
    
    def _feed_loop(self, duration_minutes):
        """Main loop for feeding live data"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while self.is_running and datetime.now() < end_time:
            try:
                # Get latest data
                data = yf.download(self.ticker, period='1d', interval=self.interval)
                if not data.empty:
                    latest_price = float(data['Close'].iloc[-1])
                    timestamp = datetime.now()
                    
                    if self.callback:
                        self.callback(timestamp, latest_price)
                
                # Wait for next interval
                time.sleep(60 if self.interval == '1m' else 300)  # 1min or 5min
                
            except Exception as e:
                print(f"Error in data feed: {e}")
                time.sleep(30)  # Wait before retry


def create_performance_summary(portfolio_values, trades, initial_balance):
    """Create a performance summary visualization"""
    if not portfolio_values or not trades:
        return None
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100
    
    completed_trades = [t for t in trades if t.get('profit', 0) != 0]
    if completed_trades:
        profits = [t['profit'] for t in completed_trades]
        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
        avg_profit = np.mean(profits)
        max_profit = max(profits)
        max_loss = min(profits)
    else:
        win_rate = avg_profit = max_profit = max_loss = 0
    
    # Create summary figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio Growth',
            'Performance Metrics',
            'Trade Distribution',
            'Cumulative Returns'
        ),
        specs=[[{"secondary_y": False}, {"type": "table"}],
               [{"type": "histogram"}, {"secondary_y": False}]]
    )
    
    # Portfolio growth
    time_range = list(range(len(portfolio_values)))
    fig.add_trace(
        go.Scatter(
            x=time_range,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=3)
        ),
        row=1, col=1
    )
    
    # Performance metrics table
    metrics = [
        ['Initial Balance', f'${initial_balance:.2f}'],
        ['Final Value', f'${final_value:.2f}'],
        ['Total Return', f'{total_return:.1f}%'],
        ['Total Trades', str(len(completed_trades))],
        ['Win Rate', f'{win_rate:.1f}%'],
        ['Avg Profit/Trade', f'${avg_profit:.2f}'],
        ['Max Profit', f'${max_profit:.2f}'],
        ['Max Loss', f'${max_loss:.2f}']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=list(zip(*metrics)))
        ),
        row=1, col=2
    )
    
    # Trade distribution
    if completed_trades:
        profits = [t['profit'] for t in completed_trades]
        fig.add_trace(
            go.Histogram(
                x=profits,
                nbinsx=20,
                name='Trade P&L',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
    
    # Cumulative returns
    returns = [(v - initial_balance) / initial_balance * 100 for v in portfolio_values]
    fig.add_trace(
        go.Scatter(
            x=time_range,
            y=returns,
            mode='lines',
            name='Cumulative Return %',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Trading Performance Summary',
        height=800,
        showlegend=True
    )
    
    return fig


# Example usage function
def demo_realtime_visualization():
    """Demonstrate real-time visualization capabilities"""
    print("Starting real-time visualization demo...")
    
    visualizer = RealTimeVisualizer(max_points=50)
    
    # Simulate some trading data
    base_price = 150.0
    portfolio_value = 10000.0
    
    for i in range(30):
        timestamp = datetime.now() - timedelta(minutes=30-i)
        
        # Simulate price movement
        price_change = np.random.normal(0, 2)
        price = base_price + price_change
        base_price = price
        
        # Simulate portfolio changes
        if i % 10 == 5:  # Buy signal
            visualizer.add_trade(timestamp, price, 'buy')
            portfolio_value -= 1000  # Use cash to buy
        elif i % 10 == 8:  # Sell signal
            profit = np.random.normal(50, 100)
            visualizer.add_trade(timestamp, price, 'sell', profit)
            portfolio_value += 1000 + profit  # Sell and realize profit
        
        # Add some noise to portfolio value
        portfolio_noise = np.random.normal(0, 50)
        current_portfolio = portfolio_value + portfolio_noise
        
        # Simulate prediction
        prediction = price + np.random.normal(0, 1)
        
        visualizer.update_data(timestamp, price, current_portfolio, prediction)
    
    # Create and save dashboard
    dashboard_file = visualizer.save_dashboard("DEMO")
    print(f"Demo dashboard saved as: {dashboard_file}")
    
    return visualizer


if __name__ == "__main__":
    # Run demo
    demo_realtime_visualization()
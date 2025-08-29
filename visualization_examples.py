#!/usr/bin/env python3
"""
Visualization Examples for ML Trading Bot

This script demonstrates how to use the visualization features added to the trading bot.
Run this script to see examples of all available visualizations.

Usage:
    python visualization_examples.py
"""

import sys
import os
import json
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.nn.model import GRUPredictor
from src.nn.data_handler import make_dataset
from src.nn.train import train_gru_predictor
from src.nn.trade import simulate_trading
from src.nn.visualizations import (
    plot_training_loss, 
    plot_portfolio_performance, 
    plot_price_predictions,
    plot_technical_indicators,
    plot_performance_metrics,
    create_interactive_dashboard
)
from src.nn.realtime_viz import demo_realtime_visualization


def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def example_basic_visualizations():
    """Demonstrate basic static visualizations"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Static Visualizations")
    print("="*60)
    
    # Load configuration
    config = load_config('src/nn/test_nn.json')
    
    # Create dataset with technical indicators visualization
    print("\n1. Creating dataset with technical indicators visualization...")
    dataset, input_dim = make_dataset(
        ticker=config['tickers'][0],
        start=config['start'],
        end=config['end'],
        seq_length=config['seq_length'],
        interval=config['interval'],
        normalize=True,
        plot_indicators=True  # This will show technical indicators
    )
    
    # Create and train model
    print("\n2. Training model with loss visualization...")
    model = GRUPredictor(input_dim=input_dim, hidden_dim=128, output_dim=1, num_layers=3)
    
    # Train with visualization
    losses = train_gru_predictor(
        model, 
        dataset, 
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        plot_loss=True  # This will show training loss
    )
    
    print(f"Training completed. Final loss: {losses[-1]:.4f}")


def example_trading_simulation():
    """Demonstrate trading simulation with visualizations"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Trading Simulation with Visualizations")
    print("="*60)
    
    # Load configuration
    config = load_config('src/nn/test_nn.json')
    
    # Create dataset (without plots to avoid repetition)
    print("\n1. Creating dataset for trading simulation...")
    dataset, input_dim = make_dataset(
        ticker=config['tickers'][0],
        start=config['start'],
        end=config['end'],
        seq_length=config['seq_length'],
        interval=config['interval'],
        normalize=True,
        plot_indicators=False  # Skip plots for this example
    )
    
    # Create and train model
    print("\n2. Training model...")
    model = GRUPredictor(input_dim=input_dim, hidden_dim=128, output_dim=1, num_layers=3)
    
    losses = train_gru_predictor(
        model, 
        dataset, 
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        plot_loss=False  # Skip plot for this example
    )
    
    # Get price data for simulation
    from src.nn.data_handler import DataLoader
    dl = DataLoader(config['tickers'][0], config['start'], config['end'], config['interval'])
    data = dl.download()[config['tickers'][0]]
    prices = data['Close'].values
    
    # Run trading simulation with visualizations
    print("\n3. Running trading simulation with visualizations...")
    portfolio_values, trades = simulate_trading(
        model=model,
        dataset=dataset,
        prices=prices,
        plot_results=True,  # This will show all trading visualizations
        initial_balance=10000
    )
    
    print(f"Simulation completed. Final portfolio value: ${portfolio_values[-1]:.2f}")
    print(f"Total trades executed: {len([t for t in trades if t['action'] in ['buy', 'sell']])}")


def example_interactive_dashboard():
    """Demonstrate interactive dashboard creation"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Interactive Dashboard")
    print("="*60)
    
    # Load configuration  
    config = load_config('src/nn/test_nn.json')
    ticker = config['tickers'][0]
    
    # Create dataset
    print("\n1. Preparing data for interactive dashboard...")
    dataset, input_dim = make_dataset(
        ticker=ticker,
        start=config['start'],
        end=config['end'],
        seq_length=config['seq_length'],
        interval=config['interval'],
        normalize=True,
        plot_indicators=False
    )
    
    # Create and train model
    print("\n2. Training model...")
    model = GRUPredictor(input_dim=input_dim, hidden_dim=128, output_dim=1, num_layers=3)
    
    train_gru_predictor(
        model, 
        dataset, 
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        plot_loss=False
    )
    
    # Get data and run simulation
    from src.nn.data_handler import DataLoader
    from src.nn.utils import add_technical_indicators
    
    dl = DataLoader(ticker, config['start'], config['end'], config['interval'])
    data = dl.download()[ticker]
    data = add_technical_indicators(data)
    prices = data['Close'].values
    
    portfolio_values, trades = simulate_trading(
        model=model,
        dataset=dataset,
        prices=prices,
        plot_results=False,  # We'll create our own dashboard
        initial_balance=10000
    )
    
    # Create interactive dashboard
    print("\n3. Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(data, portfolio_values, trades, ticker)
    
    # Save dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{ticker}_interactive_dashboard_{timestamp}.html"
    dashboard.write_html(filename)
    print(f"Interactive dashboard saved as: {filename}")
    print("Open this file in your web browser to explore the interactive dashboard!")


def example_realtime_demo():
    """Demonstrate real-time visualization capabilities"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Real-time Visualization Demo")
    print("="*60)
    
    print("\n1. Running real-time visualization demo...")
    print("This creates a simulated real-time trading environment...")
    
    # Run the demo
    visualizer = demo_realtime_visualization()
    
    print("\nReal-time demo completed!")
    print("Check the generated HTML file to see the live dashboard.")


def main():
    """Run all visualization examples"""
    print("ML Trading Bot - Visualization Examples")
    print("This script demonstrates all available visualization features.")
    print("\nNote: This will generate multiple plots and HTML files.")
    print("Press Ctrl+C to stop at any time.")
    
    try:
        # Check if required files exist
        if not os.path.exists('src/nn/test_nn.json'):
            print("Error: src/nn/test_nn.json not found. Please ensure the configuration file exists.")
            return
        
        # Run examples
        example_basic_visualizations()
        
        input("\nPress Enter to continue to trading simulation example...")
        example_trading_simulation()
        
        input("\nPress Enter to continue to interactive dashboard example...")
        example_interactive_dashboard()
        
        input("\nPress Enter to continue to real-time demo...")
        example_realtime_demo()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED!")
        print("="*60)
        print("\nGenerated files:")
        print("- Training loss plots (displayed)")
        print("- Technical indicators plots (displayed)")
        print("- Portfolio performance plots (displayed)")
        print("- Price prediction plots (displayed)")
        print("- Performance metrics plots (displayed)")
        print("- Interactive dashboard HTML files")
        print("- Real-time demo HTML file")
        print("\nYou can now integrate these visualizations into your trading workflow!")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Please ensure all dependencies are installed: pip install -r src/nn/requirements.txt")


if __name__ == "__main__":
    main()
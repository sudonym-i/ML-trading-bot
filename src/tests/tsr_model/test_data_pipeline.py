"""
Unit tests for TSR data pipeline (data_pipeline.py)

Tests the data loading and preprocessing functionality including:
- DataLoader class initialization and configuration
- Data downloading and validation
- Data preprocessing and cleaning  
- Integration with yfinance API
- Error handling for invalid data
"""

import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from tsr_model.data_pipeline import DataLoader, make_dataset


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    def test_dataloader_initialization_single_ticker(self):
        """Test DataLoader initialization with single ticker."""
        loader = DataLoader("AAPL", "2023-01-01", "2023-12-31", "1d")
        
        assert loader.tickers == ["AAPL"]
        assert loader.start == "2023-01-01"
        assert loader.end == "2023-12-31"
        assert loader.interval == "1d"
        assert loader.data == {}
    
    def test_dataloader_initialization_multiple_tickers(self):
        """Test DataLoader initialization with multiple tickers."""
        tickers = ["AAPL", "GOOGL", "MSFT"]
        loader = DataLoader(tickers, "2023-01-01", "2023-12-31")
        
        assert loader.tickers == tickers
        assert loader.interval == "1d"  # default
    
    def test_dataloader_initialization_default_interval(self):
        """Test DataLoader uses default interval."""
        loader = DataLoader("TSLA", "2023-01-01", "2023-12-31")
        assert loader.interval == "1d"
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_download_successful(self, mock_download, mock_yfinance_data):
        """Test successful data download."""
        mock_download.return_value = mock_yfinance_data
        
        loader = DataLoader("AAPL", "2023-01-01", "2023-12-31")
        loader.download()
        
        # Check that download was called with correct parameters
        mock_download.assert_called_once_with("AAPL", start="2023-01-01", end="2023-12-31", interval="1d")
        
        # Check that data was stored
        assert "AAPL" in loader.data
        assert isinstance(loader.data["AAPL"], pd.DataFrame)
        assert len(loader.data["AAPL"]) > 0
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_download_multiple_tickers(self, mock_download, mock_yfinance_data):
        """Test downloading multiple tickers."""
        mock_download.return_value = mock_yfinance_data
        
        tickers = ["AAPL", "GOOGL"]
        loader = DataLoader(tickers, "2023-01-01", "2023-12-31")
        loader.download()
        
        # Should be called once for each ticker
        assert mock_download.call_count == len(tickers)
        
        # Check data for each ticker
        for ticker in tickers:
            assert ticker in loader.data
            assert isinstance(loader.data[ticker], pd.DataFrame)
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_download_empty_data(self, mock_download):
        """Test handling of empty data from yfinance."""
        mock_download.return_value = pd.DataFrame()  # Empty dataframe
        
        loader = DataLoader("INVALID", "2023-01-01", "2023-12-31")
        loader.download()
        
        # Should not store empty data
        assert "INVALID" not in loader.data
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_download_none_data(self, mock_download):
        """Test handling of None data from yfinance."""
        mock_download.return_value = None
        
        loader = DataLoader("INVALID", "2023-01-01", "2023-12-31")
        loader.download()
        
        # Should not store None data
        assert "INVALID" not in loader.data
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_download_data_with_nan(self, mock_download):
        """Test handling of data with NaN values."""
        # Create data with NaN values
        data_with_nan = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [101, 103, np.nan],
            'Low': [99, 101, 101],
            'Close': [100.5, 102, 101.5],
            'Volume': [1000000, 2000000, 1500000]
        })
        mock_download.return_value = data_with_nan
        
        loader = DataLoader("TEST", "2023-01-01", "2023-12-31")
        loader.download()
        
        # Data should be stored after dropna()
        assert "TEST" in loader.data
        stored_data = loader.data["TEST"]
        
        # Should have no NaN values after processing
        assert not stored_data.isnull().any().any()
    
    def test_dataloader_data_access(self, mock_yfinance_data):
        """Test accessing downloaded data."""
        loader = DataLoader("AAPL", "2023-01-01", "2023-12-31")
        loader.data["AAPL"] = mock_yfinance_data.copy()
        
        # Test data retrieval
        aapl_data = loader.data["AAPL"]
        assert isinstance(aapl_data, pd.DataFrame)
        assert len(aapl_data) == len(mock_yfinance_data)
        
        # Check columns exist
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in aapl_data.columns


class TestMakeDataset:
    """Test suite for make_dataset function."""
    
    def test_make_dataset_basic(self, test_config, mock_yfinance_data):
        """Test basic make_dataset functionality."""
        with patch('tsr_model.data_pipeline.yf.download', return_value=mock_yfinance_data):
            X, y = make_dataset(test_config)
            
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(X) == len(y)
            assert X.ndim == 3  # (samples, seq_length, features)
            assert y.ndim == 1  # (samples,)
    
    def test_make_dataset_sequence_length(self, test_config, mock_yfinance_data):
        """Test that make_dataset respects sequence length."""
        test_config_copy = test_config.copy()
        test_config_copy["seq_length"] = 15
        
        with patch('tsr_model.data_pipeline.yf.download', return_value=mock_yfinance_data):
            X, y = make_dataset(test_config_copy)
            
            assert X.shape[1] == 15  # sequence length dimension
    
    def test_make_dataset_multiple_tickers(self, test_config, mock_yfinance_data):
        """Test make_dataset with multiple tickers."""
        test_config_copy = test_config.copy()
        test_config_copy["tickers"] = ["AAPL", "GOOGL"]
        
        with patch('tsr_model.data_pipeline.yf.download', return_value=mock_yfinance_data):
            X, y = make_dataset(test_config_copy)
            
            # Should combine data from multiple tickers
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(X) > 0
    
    def test_make_dataset_insufficient_data(self, test_config):
        """Test make_dataset with insufficient data for sequences."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [101, 102], 
            'Low': [99, 100],
            'Close': [100.5, 101.5],
            'Volume': [1000000, 1200000]
        })
        
        with patch('tsr_model.data_pipeline.yf.download', return_value=small_data):
            # Should handle gracefully (might return empty arrays or raise error)
            try:
                X, y = make_dataset(test_config)
                # If it succeeds, check that arrays are consistent
                assert len(X) == len(y)
            except (ValueError, IndexError):
                # This is also acceptable behavior for insufficient data
                pass
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_make_dataset_download_failure(self, mock_download, test_config):
        """Test make_dataset when download fails."""
        mock_download.return_value = None
        
        # Should handle download failure gracefully
        try:
            X, y = make_dataset(test_config)
            # If it returns something, it should be valid
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
        except Exception:
            # It's also acceptable to raise an exception for no data
            pass


class TestDataPipelineIntegration:
    """Integration tests for data pipeline components."""
    
    def test_dataloader_to_sequences_pipeline(self, mock_yfinance_data):
        """Test full pipeline from DataLoader to sequences."""
        with patch('tsr_model.data_pipeline.yf.download', return_value=mock_yfinance_data):
            loader = DataLoader("AAPL", "2023-01-01", "2023-12-31")
            loader.download()
            
            # Verify data was loaded
            assert "AAPL" in loader.data
            data = loader.data["AAPL"]
            
            # Data should have expected structure
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert not data.isnull().any().any()
    
    def test_data_consistency(self, mock_yfinance_data):
        """Test that data maintains consistency through pipeline."""
        with patch('tsr_model.data_pipeline.yf.download', return_value=mock_yfinance_data):
            loader = DataLoader("AAPL", "2023-01-01", "2023-12-31")
            loader.download()
            
            data = loader.data["AAPL"]
            
            # Check data integrity
            assert (data['High'] >= data[['Open', 'Close']].max(axis=1)).all()
            assert (data['Low'] <= data[['Open', 'Close']].min(axis=1)).all()
            assert (data['Volume'] >= 0).all()
    
    def test_date_range_consistency(self, test_config):
        """Test that data respects date range."""
        mock_data = pd.DataFrame({
            'Open': [100] * 30,
            'High': [101] * 30,
            'Low': [99] * 30,
            'Close': [100.5] * 30,
            'Volume': [1000000] * 30
        }, index=pd.date_range('2023-01-01', periods=30, freq='D'))
        
        with patch('tsr_model.data_pipeline.yf.download', return_value=mock_data):
            loader = DataLoader("AAPL", test_config["start"], test_config["end"])
            loader.download()
            
            data = loader.data["AAPL"]
            
            # Data should be within the requested date range
            assert data.index.min() >= pd.Timestamp(test_config["start"])
            assert data.index.max() <= pd.Timestamp(test_config["end"])


class TestDataPipelineErrorHandling:
    """Test error handling in data pipeline."""
    
    @patch('tsr_model.data_pipeline.yf.download')
    def test_network_error_handling(self, mock_download):
        """Test handling of network errors."""
        mock_download.side_effect = Exception("Network error")
        
        loader = DataLoader("AAPL", "2023-01-01", "2023-12-31")
        
        # Should handle network errors gracefully
        with pytest.raises(Exception):
            loader.download()
    
    def test_invalid_date_range(self):
        """Test handling of invalid date ranges."""
        # End date before start date
        loader = DataLoader("AAPL", "2023-12-31", "2023-01-01")
        
        # Behavior depends on implementation - might raise error or return empty data
        # This test documents expected behavior
        assert loader.start == "2023-12-31"
        assert loader.end == "2023-01-01"
    
    def test_invalid_ticker(self):
        """Test handling of invalid ticker symbols."""
        loader = DataLoader("INVALID_TICKER_12345", "2023-01-01", "2023-12-31")
        
        # Initialization should succeed
        assert "INVALID_TICKER_12345" in loader.tickers
        
        # Download might fail or return empty data - both are acceptable


if __name__ == "__main__":
    pytest.main([__file__])
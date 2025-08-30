"""
Unit tests for TSR utility functions (utils.py)

Tests the utility functions including:
- Technical indicator calculations (SMA, RSI, MACD)
- Sequence creation for time series data
- Data preprocessing and validation
- Mathematical correctness of indicators
"""

import pytest
import pandas as pd
import numpy as np
from tsr_model.utils import add_technical_indicators, create_sequences


class TestAddTechnicalIndicators:
    """Test suite for technical indicator calculations."""
    
    def test_add_indicators_basic(self, sample_stock_data):
        """Test basic technical indicator addition."""
        df_with_indicators = add_technical_indicators(sample_stock_data.copy())
        
        # Check that new columns were added
        expected_columns = ['SMA_14', 'RSI_14', 'MACD']
        for col in expected_columns:
            assert col in df_with_indicators.columns
        
        # Check that original columns are preserved
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in original_columns:
            assert col in df_with_indicators.columns
    
    def test_sma_calculation(self, sample_stock_data):
        """Test Simple Moving Average calculation."""
        df = sample_stock_data.copy()
        df_with_indicators = add_technical_indicators(df)
        
        # Check SMA_14 calculation manually for a few points
        sma_manual = df['Close'].rolling(window=14).mean()
        sma_from_function = df_with_indicators['SMA_14']
        
        # Should be equal where both have values (after dropna)
        valid_indices = ~(sma_manual.isna() | sma_from_function.isna())
        if valid_indices.any():
            pd.testing.assert_series_equal(
                sma_manual[valid_indices], 
                sma_from_function[valid_indices],
                check_names=False
            )
    
    def test_rsi_calculation_bounds(self, sample_stock_data):
        """Test that RSI values are within valid bounds (0-100)."""
        df_with_indicators = add_technical_indicators(sample_stock_data.copy())
        
        rsi_values = df_with_indicators['RSI_14'].dropna()
        
        # RSI should be between 0 and 100
        assert (rsi_values >= 0).all(), "RSI values below 0 detected"
        assert (rsi_values <= 100).all(), "RSI values above 100 detected"
    
    def test_rsi_calculation_logic(self):
        """Test RSI calculation with known values."""
        # Create test data with known price movements
        prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                  46.03, 46.83, 46.69, 46.45, 46.59, 46.3, 46.02, 46.72, 46.07, 45.45]
        
        df = pd.DataFrame({'Close': prices})
        df_with_indicators = add_technical_indicators(df)
        
        rsi_values = df_with_indicators['RSI_14'].dropna()
        
        # RSI should exist and be reasonable
        assert len(rsi_values) > 0
        assert rsi_values.iloc[-1] > 0
        assert rsi_values.iloc[-1] < 100
    
    def test_macd_calculation(self, sample_stock_data):
        """Test MACD calculation."""
        df_with_indicators = add_technical_indicators(sample_stock_data.copy())
        
        macd_values = df_with_indicators['MACD'].dropna()
        
        # MACD should exist and can be positive or negative
        assert len(macd_values) > 0
        assert not macd_values.isna().any()
        
        # MACD is difference of EMAs, so verify manually
        df = sample_stock_data.copy()
        ema12_manual = df['Close'].ewm(span=12, adjust=False).mean()
        ema26_manual = df['Close'].ewm(span=26, adjust=False).mean()
        macd_manual = ema12_manual - ema26_manual
        
        # Compare with function output (after dropna)
        valid_indices = ~(macd_manual.isna() | df_with_indicators['MACD'].isna())
        if valid_indices.any():
            np.testing.assert_array_almost_equal(
                macd_manual[valid_indices].values,
                df_with_indicators['MACD'][valid_indices].values,
                decimal=10
            )
    
    def test_indicators_with_insufficient_data(self):
        """Test technical indicators with insufficient data."""
        # Create data with less than 14 periods
        small_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        df_with_indicators = add_technical_indicators(small_df)
        
        # Should handle gracefully - might return empty dataframe after dropna
        # or have NaN values for indicators requiring more data
        assert isinstance(df_with_indicators, pd.DataFrame)
    
    def test_indicators_with_constant_prices(self):
        """Test technical indicators with constant prices."""
        # All prices the same
        constant_df = pd.DataFrame({
            'Open': [100] * 20,
            'High': [100] * 20,
            'Low': [100] * 20,
            'Close': [100] * 20,
            'Volume': [1000000] * 20
        })
        
        df_with_indicators = add_technical_indicators(constant_df)
        
        # SMA should equal the constant price
        assert (df_with_indicators['SMA_14'].dropna() == 100).all()
        
        # RSI should be around 50 (no consistent up/down movement)
        rsi_values = df_with_indicators['RSI_14'].dropna()
        if len(rsi_values) > 0:
            # With constant prices, RSI calculation might produce NaN or 50
            assert rsi_values.isna().all() or np.isclose(rsi_values, 50, atol=1).all()
        
        # MACD should be 0 (EMAs of constant value are constant)
        macd_values = df_with_indicators['MACD'].dropna()
        if len(macd_values) > 0:
            np.testing.assert_array_almost_equal(macd_values.values, 0, decimal=10)
    
    def test_dropna_behavior(self, sample_stock_data):
        """Test that dropna removes rows with NaN indicators."""
        df_with_indicators = add_technical_indicators(sample_stock_data.copy())
        
        # Should not have any NaN values after processing
        assert not df_with_indicators.isna().any().any()
        
        # Length should be less than original due to indicator calculation period
        assert len(df_with_indicators) <= len(sample_stock_data)


class TestCreateSequences:
    """Test suite for sequence creation functionality."""
    
    def test_create_sequences_basic(self, sample_stock_data_with_indicators):
        """Test basic sequence creation."""
        seq_length = 10
        X, y = create_sequences(sample_stock_data_with_indicators, seq_length)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 3  # (samples, seq_length, features)
        assert y.ndim == 1  # (samples,)
        
        # Check shapes
        n_samples = len(sample_stock_data_with_indicators) - seq_length
        n_features = len(sample_stock_data_with_indicators.columns)
        
        assert X.shape == (n_samples, seq_length, n_features)
        assert y.shape == (n_samples,)
    
    def test_create_sequences_different_lengths(self, sample_stock_data_with_indicators):
        """Test sequence creation with different sequence lengths."""
        for seq_length in [5, 15, 20]:
            if len(sample_stock_data_with_indicators) > seq_length:
                X, y = create_sequences(sample_stock_data_with_indicators, seq_length)
                
                expected_samples = len(sample_stock_data_with_indicators) - seq_length
                expected_features = len(sample_stock_data_with_indicators.columns)
                
                assert X.shape == (expected_samples, seq_length, expected_features)
                assert y.shape == (expected_samples,)
    
    def test_create_sequences_target_values(self, sample_stock_data_with_indicators):
        """Test that target values are correct."""
        seq_length = 5
        X, y = create_sequences(sample_stock_data_with_indicators, seq_length)
        
        # Target should be the Close price at seq_length position
        for i in range(len(y)):
            expected_target = sample_stock_data_with_indicators.iloc[i + seq_length]['Close']
            assert np.isclose(y[i], expected_target), f"Target mismatch at index {i}"
    
    def test_create_sequences_input_values(self, sample_stock_data_with_indicators):
        """Test that input sequences are correct."""
        seq_length = 3
        X, y = create_sequences(sample_stock_data_with_indicators, seq_length)
        
        if len(X) > 0:
            # Check first sequence
            first_sequence = X[0]
            expected_sequence = sample_stock_data_with_indicators.iloc[0:seq_length].values
            
            np.testing.assert_array_almost_equal(first_sequence, expected_sequence)
    
    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        # Create data smaller than sequence length
        small_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        seq_length = 10
        X, y = create_sequences(small_df, seq_length)
        
        # Should return empty arrays
        assert len(X) == 0
        assert len(y) == 0
        assert X.shape[1:] == (seq_length, len(small_df.columns))
    
    def test_create_sequences_exact_length(self):
        """Test sequence creation when data length equals sequence length."""
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        seq_length = 5
        X, y = create_sequences(df, seq_length)
        
        # Should return empty arrays (no samples can be created)
        assert len(X) == 0
        assert len(y) == 0
    
    def test_create_sequences_single_sample(self):
        """Test sequence creation that produces single sample."""
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        })
        
        seq_length = 5
        X, y = create_sequences(df, seq_length)
        
        # Should produce exactly one sample
        assert X.shape == (1, seq_length, len(df.columns))
        assert y.shape == (1,)
        assert y[0] == 105  # Last close price
    
    def test_create_sequences_data_types(self, sample_stock_data_with_indicators):
        """Test that sequences have correct data types."""
        seq_length = 10
        X, y = create_sequences(sample_stock_data_with_indicators, seq_length)
        
        assert X.dtype in [np.float32, np.float64]
        assert y.dtype in [np.float32, np.float64]
        
        # Should not have any NaN or infinite values
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        assert not np.isinf(X).any()
        assert not np.isinf(y).any()


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_indicators_to_sequences_pipeline(self, sample_stock_data):
        """Test full pipeline from raw data to sequences."""
        # Add indicators
        df_with_indicators = add_technical_indicators(sample_stock_data.copy())
        
        # Create sequences
        seq_length = 8
        X, y = create_sequences(df_with_indicators, seq_length)
        
        # Verify full pipeline
        if len(X) > 0:
            assert X.shape[0] == y.shape[0]  # Same number of samples
            assert X.shape[1] == seq_length   # Correct sequence length
            assert X.shape[2] == len(df_with_indicators.columns)  # All features included
    
    def test_multiple_processing_consistency(self, sample_stock_data):
        """Test that multiple processing runs produce consistent results."""
        # Process same data twice
        df1 = add_technical_indicators(sample_stock_data.copy())
        df2 = add_technical_indicators(sample_stock_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Sequence creation should also be consistent
        seq_length = 10
        X1, y1 = create_sequences(df1, seq_length)
        X2, y2 = create_sequences(df2, seq_length)
        
        if len(X1) > 0:
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)


class TestUtilsErrorHandling:
    """Test error handling in utility functions."""
    
    def test_indicators_empty_dataframe(self):
        """Test technical indicators with empty dataframe."""
        empty_df = pd.DataFrame()
        
        try:
            result = add_technical_indicators(empty_df)
            # If it succeeds, should return empty dataframe
            assert len(result) == 0
        except (KeyError, ValueError):
            # It's also acceptable to raise an error
            pass
    
    def test_indicators_missing_close_column(self):
        """Test technical indicators without Close column."""
        df_no_close = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        with pytest.raises(KeyError):
            add_technical_indicators(df_no_close)
    
    def test_sequences_invalid_seq_length(self, sample_stock_data_with_indicators):
        """Test sequence creation with invalid sequence length."""
        # Zero sequence length
        with pytest.raises((ValueError, AssertionError)):
            create_sequences(sample_stock_data_with_indicators, 0)
        
        # Negative sequence length  
        with pytest.raises((ValueError, AssertionError)):
            create_sequences(sample_stock_data_with_indicators, -5)


if __name__ == "__main__":
    pytest.main([__file__])
"""
Unit tests for TSR route functions (route.py)

Tests the high-level training and testing routes including:
- train_model function integration
- test_run_model function integration  
- Configuration loading and validation
- Model training pipeline
- Error handling for missing config
"""

import pytest
import torch
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from tsr_model.route import train_model, test_run_model
from tsr_model.model import GRUPredictor


class TestTrainModel:
    """Test suite for train_model function."""
    
    def test_train_model_config_loading(self, test_config):
        """Test that train_model loads configuration correctly."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.train_gru_predictor') as mock_train:
                        # Setup mock returns
                        mock_dataset = (torch.randn(50, 10, 8), torch.randn(50))
                        mock_make_dataset.return_value = mock_dataset
                        
                        model = train_model()
                        
                        # Verify config was used
                        mock_make_dataset.assert_called_once()
                        mock_train.assert_called_once()
                        
                        # Check that returned model is correct type
                        assert isinstance(model, GRUPredictor)
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_input_dimension_calculation(self, test_config):
        """Test that input dimension is calculated correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.train_gru_predictor') as mock_train:
                        # Mock dataset with specific feature dimension
                        n_features = 12
                        mock_dataset = (torch.randn(50, 10, n_features), torch.randn(50))
                        mock_make_dataset.return_value = mock_dataset
                        
                        model = train_model()
                        
                        # Check that model was created with correct input dimension
                        assert model.gru.input_size == n_features
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_missing_config_file(self):
        """Test train_model behavior with missing config file."""
        with patch('tsr_model.route.os.path.join', return_value='/nonexistent/config.json'):
            with pytest.raises(FileNotFoundError):
                train_model()
    
    def test_train_model_invalid_json(self):
        """Test train_model behavior with invalid JSON config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with pytest.raises(json.JSONDecodeError):
                    train_model()
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_missing_tsr_model_section(self):
        """Test train_model with missing tsr_model section in config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"other_model": {"training": {}}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with pytest.raises(KeyError):
                    train_model()
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_missing_training_section(self):
        """Test train_model with missing training section in config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"testing": {}}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with pytest.raises(KeyError):
                    train_model()
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_dataset_integration(self, test_config):
        """Test integration with dataset creation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.train_gru_predictor') as mock_train:
                        mock_dataset = (torch.randn(100, 24, 8), torch.randn(100))
                        mock_make_dataset.return_value = mock_dataset
                        
                        model = train_model()
                        
                        # Verify make_dataset was called with training config
                        call_args = mock_make_dataset.call_args[0][0]
                        assert call_args["tickers"] == test_config["tickers"]
                        assert call_args["start"] == test_config["start"]
                        assert call_args["end"] == test_config["end"]
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_training_parameters(self, test_config):
        """Test that training parameters are passed correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.train_gru_predictor') as mock_train:
                        mock_dataset = (torch.randn(50, 10, 8), torch.randn(50))
                        mock_make_dataset.return_value = mock_dataset
                        
                        train_model()
                        
                        # Check training parameters
                        call_args = mock_train.call_args
                        assert call_args[1]["epochs"] == test_config["epochs"]
                        assert call_args[1]["batch_size"] == test_config["batch_size"]
                        assert call_args[1]["lr"] == test_config["lr"]
        finally:
            os.unlink(temp_config_path)


class TestTestRunModel:
    """Test suite for test_run_model function."""
    
    def test_test_run_model_config_loading(self, sample_gru_model, test_config):
        """Test that test_run_model loads configuration correctly."""
        # Create config with testing section
        test_config_copy = test_config.copy()
        test_config_copy.update({
            "start": "2023-06-01",
            "end": "2023-12-31"
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"testing": test_config_copy}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.simulate_trading') as mock_simulate:
                        mock_dataset = (torch.randn(30, 10, 8), torch.randn(30))
                        mock_make_dataset.return_value = mock_dataset
                        
                        test_run_model(sample_gru_model)
                        
                        # Verify config was used
                        mock_make_dataset.assert_called_once()
                        mock_simulate.assert_called_once()
        finally:
            os.unlink(temp_config_path)
    
    def test_test_run_model_missing_testing_section(self, sample_gru_model):
        """Test test_run_model with missing testing section."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": {}}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with pytest.raises(KeyError):
                    test_run_model(sample_gru_model)
        finally:
            os.unlink(temp_config_path)
    
    def test_test_run_model_simulation_integration(self, sample_gru_model, test_config):
        """Test integration with trading simulation."""
        test_config_copy = test_config.copy()
        test_config_copy.update({
            "start": "2023-06-01", 
            "end": "2023-12-31"
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"testing": test_config_copy}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.simulate_trading') as mock_simulate:
                        mock_dataset = (torch.randn(40, 24, 8), torch.randn(40))
                        mock_make_dataset.return_value = mock_dataset
                        
                        test_run_model(sample_gru_model)
                        
                        # Verify simulate_trading was called with model and dataset
                        call_args = mock_simulate.call_args[0]
                        assert call_args[0] == sample_gru_model
                        assert len(call_args[1]) == 2  # (X, y) tuple
        finally:
            os.unlink(temp_config_path)
    
    def test_test_run_model_none_model(self, test_config):
        """Test test_run_model with None model."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"testing": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.simulate_trading') as mock_simulate:
                        mock_dataset = (torch.randn(30, 10, 8), torch.randn(30))
                        mock_make_dataset.return_value = mock_dataset
                        
                        # Should handle None model gracefully or raise appropriate error
                        try:
                            test_run_model(None)
                            # If it succeeds, simulate_trading should be called with None
                            mock_simulate.assert_called_once()
                            assert mock_simulate.call_args[0][0] is None
                        except (AttributeError, TypeError):
                            # It's also acceptable to raise an error for None model
                            pass
        finally:
            os.unlink(temp_config_path)


class TestRouteIntegration:
    """Integration tests for route functions."""
    
    def test_train_then_test_pipeline(self, test_config):
        """Test training a model then testing it."""
        # Create comprehensive config
        full_config = {
            "tsr_model": {
                "training": test_config,
                "testing": {
                    **test_config,
                    "start": "2023-06-01",
                    "end": "2023-12-31"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(full_config, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.train_gru_predictor') as mock_train:
                        with patch('tsr_model.route.simulate_trading') as mock_simulate:
                            # Setup mock returns
                            train_dataset = (torch.randn(50, 10, 8), torch.randn(50))
                            test_dataset = (torch.randn(30, 10, 8), torch.randn(30))
                            mock_make_dataset.side_effect = [train_dataset, test_dataset]
                            
                            # Train model
                            model = train_model()
                            
                            # Test model
                            test_run_model(model)
                            
                            # Verify complete pipeline
                            assert mock_make_dataset.call_count == 2
                            mock_train.assert_called_once()
                            mock_simulate.assert_called_once()
                            
                            # Check that same model instance was used
                            assert mock_simulate.call_args[0][0] == model
        finally:
            os.unlink(temp_config_path)
    
    def test_config_path_calculation(self):
        """Test that config path is calculated correctly."""
        with patch('tsr_model.route.os.path.dirname') as mock_dirname:
            with patch('tsr_model.route.os.path.join') as mock_join:
                with patch('builtins.open', side_effect=FileNotFoundError):
                    # Setup directory mocking
                    mock_dirname.return_value = '/path/to/src/tsr_model'
                    
                    try:
                        train_model()
                    except FileNotFoundError:
                        pass  # Expected
                    
                    # Check that path was constructed correctly
                    # Should go up 3 levels: tsr_model -> src -> root
                    mock_join.assert_called()
                    call_args = mock_join.call_args[0]
                    assert 'config.json' in call_args


class TestRouteErrorHandling:
    """Test error handling in route functions."""
    
    def test_train_model_dataset_creation_failure(self, test_config):
        """Test train_model when dataset creation fails."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset', side_effect=Exception("Dataset creation failed")):
                    with pytest.raises(Exception, match="Dataset creation failed"):
                        train_model()
        finally:
            os.unlink(temp_config_path)
    
    def test_train_model_training_failure(self, test_config):
        """Test train_model when training fails."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"tsr_model": {"training": test_config}}
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                    with patch('tsr_model.route.train_gru_predictor', side_effect=Exception("Training failed")):
                        mock_dataset = (torch.randn(50, 10, 8), torch.randn(50))
                        mock_make_dataset.return_value = mock_dataset
                        
                        with pytest.raises(Exception, match="Training failed"):
                            train_model()
        finally:
            os.unlink(temp_config_path)
    
    def test_config_validation(self):
        """Test config validation for required fields."""
        incomplete_configs = [
            # Missing tickers
            {"start": "2023-01-01", "end": "2023-12-31", "seq_length": 10},
            # Missing dates
            {"tickers": ["AAPL"], "seq_length": 10},
            # Missing seq_length
            {"tickers": ["AAPL"], "start": "2023-01-01", "end": "2023-12-31"}
        ]
        
        for incomplete_config in incomplete_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_data = {"tsr_model": {"training": incomplete_config}}
                json.dump(config_data, f)
                temp_config_path = f.name
            
            try:
                with patch('tsr_model.route.os.path.join', return_value=temp_config_path):
                    with patch('tsr_model.route.make_dataset') as mock_make_dataset:
                        # make_dataset should be called and might handle missing fields
                        mock_make_dataset.side_effect = KeyError("Missing required field")
                        
                        with pytest.raises(KeyError):
                            train_model()
            finally:
                os.unlink(temp_config_path)


if __name__ == "__main__":
    pytest.main([__file__])
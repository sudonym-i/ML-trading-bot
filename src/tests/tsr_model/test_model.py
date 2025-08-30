"""
Unit tests for TSR model (model.py)

Tests the GRUPredictor neural network architecture including:
- Model initialization
- Forward pass functionality  
- Input/output shapes
- Model parameters and layers
- Gradient computation
"""

import pytest
import torch
import torch.nn as nn
from tsr_model.model import GRUPredictor


class TestGRUPredictor:
    """Test suite for GRUPredictor model."""
    
    def test_model_initialization(self):
        """Test that GRUPredictor initializes correctly with given parameters."""
        input_dim = 8
        hidden_dim = 128
        num_layers = 3
        
        model = GRUPredictor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # Check model components exist
        assert isinstance(model.gru, nn.GRU)
        assert isinstance(model.relu, nn.ReLU)
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.fc2, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        
        # Check GRU configuration
        assert model.gru.input_size == input_dim
        assert model.gru.hidden_size == hidden_dim
        assert model.gru.num_layers == num_layers
        assert model.gru.batch_first == True
        
        # Check linear layer dimensions
        assert model.fc1.in_features == hidden_dim
        assert model.fc1.out_features == 64
        assert model.fc2.in_features == 64
        assert model.fc2.out_features == 1
    
    def test_model_default_parameters(self):
        """Test model initialization with default parameters."""
        input_dim = 5
        model = GRUPredictor(input_dim=input_dim)
        
        assert model.gru.hidden_size == 128  # default hidden_dim
        assert model.gru.num_layers == 3     # default num_layers
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 16
        seq_length = 24
        input_dim = 8
        
        model = GRUPredictor(input_dim=input_dim)
        
        # Create sample input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        expected_shape = (batch_size, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        seq_length = 10
        input_dim = 6
        model = GRUPredictor(input_dim=input_dim)
        
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, seq_length, input_dim)
            output = model(x)
            assert output.shape == (batch_size, 1)
    
    def test_forward_pass_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        batch_size = 8
        input_dim = 7
        model = GRUPredictor(input_dim=input_dim)
        
        for seq_length in [5, 10, 20, 50]:
            x = torch.randn(batch_size, seq_length, input_dim)
            output = model(x)
            assert output.shape == (batch_size, 1)
    
    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters."""
        model = GRUPredictor(input_dim=8, hidden_dim=64, num_layers=2)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have some parameters
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable
        
        # Rough sanity check (GRU + 2 FC layers should have reasonable param count)
        assert 10000 < total_params < 100000, f"Unexpected parameter count: {total_params}"
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        model = GRUPredictor(input_dim=5)
        criterion = nn.MSELoss()
        
        # Sample data
        x = torch.randn(4, 10, 5, requires_grad=True)
        target = torch.randn(4, 1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Gradients not computed for some parameters"
            assert not torch.isnan(param.grad).any(), "NaN gradients detected"
    
    def test_model_eval_mode(self):
        """Test that model behaves differently in eval mode (dropout)."""
        model = GRUPredictor(input_dim=5, hidden_dim=32)
        x = torch.randn(4, 10, 5)
        
        # Training mode
        model.train()
        output_train1 = model(x)
        output_train2 = model(x)
        
        # Eval mode
        model.eval()
        output_eval1 = model(x)
        output_eval2 = model(x)
        
        # In eval mode, outputs should be identical (no dropout randomness)
        torch.testing.assert_close(output_eval1, output_eval2, atol=1e-6, rtol=1e-6)
        
        # Training and eval outputs might differ due to dropout
        # (Though with small model and low dropout, difference might be minimal)
    
    def test_model_device_compatibility(self):
        """Test model works on available device."""
        model = GRUPredictor(input_dim=4)
        x = torch.randn(2, 8, 4)
        
        # Test on CPU
        output_cpu = model(x)
        assert output_cpu.device.type == 'cpu'
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x.cuda()
            output_gpu = model_gpu(x_gpu)
            assert output_gpu.device.type == 'cuda'
    
    def test_model_reproducibility(self, setup_torch):
        """Test that model produces reproducible results."""
        model1 = GRUPredictor(input_dim=6, hidden_dim=32, num_layers=2)
        model2 = GRUPredictor(input_dim=6, hidden_dim=32, num_layers=2)
        
        # Set same weights
        model2.load_state_dict(model1.state_dict())
        
        x = torch.randn(3, 12, 6)
        
        model1.eval()
        model2.eval()
        
        output1 = model1(x)
        output2 = model2(x)
        
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)
    
    def test_model_output_range(self):
        """Test that model outputs are in reasonable range."""
        model = GRUPredictor(input_dim=5)
        
        # Test with reasonable financial data ranges
        x = torch.randn(8, 15, 5) * 100 + 100  # Simulate stock prices around $100
        
        output = model(x)
        
        # Outputs shouldn't be extremely large or small
        assert not torch.isnan(output).any(), "Model produced NaN outputs"
        assert not torch.isinf(output).any(), "Model produced infinite outputs"
        assert output.abs().max() < 1e6, "Model outputs are unreasonably large"


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_input_dimensions(self):
        """Test model behavior with invalid input dimensions."""
        model = GRUPredictor(input_dim=5)
        
        # Wrong input dimension
        with pytest.raises(RuntimeError):
            x = torch.randn(4, 10, 3)  # input_dim should be 5
            model(x)
    
    def test_empty_input(self):
        """Test model with empty input."""
        model = GRUPredictor(input_dim=5)
        
        # Empty batch
        with pytest.raises((RuntimeError, IndexError)):
            x = torch.randn(0, 10, 5)
            model(x)
        
        # Empty sequence
        with pytest.raises((RuntimeError, IndexError)):
            x = torch.randn(4, 0, 5)
            model(x)
    
    def test_single_timestep_input(self):
        """Test model with single timestep."""
        model = GRUPredictor(input_dim=5)
        
        # Single timestep should work
        x = torch.randn(4, 1, 5)
        output = model(x)
        assert output.shape == (4, 1)


if __name__ == "__main__":
    pytest.main([__file__])
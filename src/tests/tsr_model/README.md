# TSR Model Test Suite

Comprehensive unit tests for the Time Series Regression (TSR) model components of the ML Trading Bot.

## Overview

This test suite provides comprehensive coverage for all TSR model components:

- **`test_model.py`**: Tests for `GRUPredictor` neural network model
- **`test_data_pipeline.py`**: Tests for data loading and preprocessing  
- **`test_utils.py`**: Tests for utility functions (technical indicators, sequences)
- **`test_route.py`**: Tests for training and testing routes
- **`conftest.py`**: Shared fixtures and test configuration

## Quick Start

### Run All Tests
```bash
# From the test directory
cd src/tests/tsr_model
python -m pytest

# Or use the test runner
python run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only  
python run_tests.py --integration

# Include slow tests
python run_tests.py --slow

# Run with coverage
python run_tests.py --coverage
```

### Run Specific Test Files
```bash
# Test only the model
python -m pytest test_model.py

# Test only utilities
python -m pytest test_utils.py

# Run specific test function
python -m pytest test_model.py::TestGRUPredictor::test_model_initialization
```

## Test Structure

### Test Categories

Tests are organized with pytest markers:

- **`@pytest.mark.unit`**: Fast unit tests for individual components
- **`@pytest.mark.integration`**: Integration tests across components  
- **`@pytest.mark.slow`**: Tests that take longer to run
- **`@pytest.mark.network`**: Tests requiring network access (yfinance)

### Test Fixtures

The `conftest.py` file provides shared fixtures:

- **`sample_stock_data`**: Realistic mock stock price data
- **`sample_stock_data_with_indicators`**: Stock data with technical indicators
- **`sample_sequences`**: Prepared sequence data for model training
- **`sample_gru_model`**: Pre-initialized GRU model for testing
- **`test_config`**: Standard test configuration dictionary
- **`setup_torch`**: Ensures deterministic PyTorch behavior

## Test Coverage

### Model Tests (`test_model.py`)
- ✅ Model initialization with different parameters
- ✅ Forward pass with various input shapes  
- ✅ Gradient computation and backpropagation
- ✅ Training/evaluation mode differences
- ✅ Device compatibility (CPU/GPU)
- ✅ Parameter counting and validation
- ✅ Edge cases and error handling

### Data Pipeline Tests (`test_data_pipeline.py`)  
- ✅ DataLoader initialization and configuration
- ✅ yfinance API integration (mocked)
- ✅ Data validation and cleaning
- ✅ Multi-ticker data handling
- ✅ Error handling for network issues
- ✅ Date range validation
- ✅ Empty/invalid data handling

### Utils Tests (`test_utils.py`)
- ✅ Technical indicator calculations (SMA, RSI, MACD)
- ✅ Mathematical correctness of indicators
- ✅ Sequence creation for time series data
- ✅ Input/output shape validation
- ✅ Edge cases (insufficient data, constant prices)
- ✅ Data type consistency

### Route Tests (`test_route.py`)
- ✅ Configuration loading and parsing
- ✅ Training pipeline integration
- ✅ Testing pipeline integration  
- ✅ Error handling for missing config
- ✅ Parameter passing validation
- ✅ End-to-end workflow testing

## Running Tests

### Prerequisites

Install test dependencies:
```bash
# Basic testing
pip install pytest

# Additional testing tools (optional)
pip install pytest-cov pytest-xdist pytest-timeout
```

### Test Configuration

The test suite uses `pytest.ini` for configuration:

- **Test discovery**: Finds all `test_*.py` files
- **Output formatting**: Verbose output with short tracebacks
- **Markers**: Defined test categories
- **Timeouts**: 5-minute timeout for long tests
- **Warnings**: Filters out common library warnings

### Environment Variables

Tests respect these environment variables:

- **`SKIP_NETWORK_TESTS=1`**: Skip tests requiring network access
- **`TORCH_HOME=/tmp/torch_cache`**: PyTorch model cache location
- **`PYTHONPATH=../../..`**: Ensures proper module imports

### Coverage Reports

Generate coverage reports:
```bash
# Terminal coverage report
python run_tests.py --coverage

# HTML coverage report
python run_tests.py --html-cov

# Open HTML report
open htmlcov/index.html
```

## Test Development

### Adding New Tests

1. **Create test file**: Follow `test_*.py` naming convention
2. **Import fixtures**: Use shared fixtures from `conftest.py`
3. **Add markers**: Tag tests with appropriate markers
4. **Mock external dependencies**: Use `unittest.mock` for yfinance, etc.
5. **Test edge cases**: Include error conditions and boundary cases

### Example Test Function
```python
import pytest
from tsr_model.model import GRUPredictor

class TestNewFeature:
    def test_feature_basic_functionality(self, sample_stock_data):
        """Test basic feature functionality."""
        # Setup
        model = GRUPredictor(input_dim=8)
        
        # Execute
        result = model.new_feature(sample_stock_data)
        
        # Verify
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.slow
    def test_feature_with_large_dataset(self, large_dataset):
        """Test feature with large dataset - marked as slow."""
        # Test implementation
        pass
        
    @pytest.mark.network  
    def test_feature_with_live_data(self):
        """Test feature with live data - marked as network."""
        # Test implementation
        pass
```

### Best Practices

1. **Test isolation**: Each test should be independent
2. **Deterministic**: Use fixed random seeds for reproducible results
3. **Fast execution**: Keep unit tests under 1 second each
4. **Clear assertions**: Use descriptive assertion messages
5. **Mock external dependencies**: Don't rely on network/filesystem
6. **Test edge cases**: Include error conditions and boundary values

## Continuous Integration

The test suite is designed for CI/CD integration:

```bash
# CI test command
python run_tests.py --coverage --quiet --tb=line

# Fast test subset (exclude slow/network tests)  
python run_tests.py --unit -q
```

## Troubleshooting

### Common Issues

**Import Errors**: 
```bash
# Set PYTHONPATH
export PYTHONPATH=../../..
python -m pytest
```

**Fixture Not Found**:
- Check `conftest.py` is in the test directory
- Verify fixture name spelling

**Network Test Failures**:
```bash
# Skip network tests
python run_tests.py --unit
```

**GPU Test Failures**:
```bash  
# Run CPU-only tests
CUDA_VISIBLE_DEVICES="" python -m pytest
```

### Debug Mode
```bash
# Drop into debugger on failure
python -m pytest --pdb

# Run last failed tests
python -m pytest --lf

# Extra verbose output
python -m pytest -vvv
```

## Contributing

When adding new TSR model features:

1. **Write tests first** (TDD approach)
2. **Test both success and failure cases**
3. **Add appropriate test markers**
4. **Update this README** if needed
5. **Ensure all tests pass** before committing

The test suite ensures the TSR model remains reliable and maintainable as the codebase evolves.
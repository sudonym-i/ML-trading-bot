"""
Sentiment Model Package

This package contains components for sentiment analysis of financial text data,
including web scraping, text processing, tokenization, and sentiment classification.

Modules:
    tokenize_pipeline: Text tokenization and preprocessing pipeline
    classify_model: Sentiment classification models
    web_scraper: C++ web scraping utilities (accessed via build system)
    
Author: ML Trading Bot Project
"""

__version__ = "1.0.0"

# Import available Python modules
try:
    from .tokenize_pipeline import TokenizationPipeline, TokenizationConfig
except ImportError:
    pass

__all__ = [
    "TokenizationPipeline",
    "TokenizationConfig",
]
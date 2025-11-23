"""
Pytest Configuration and Shared Fixtures
=========================================
Provides reusable test data and mock objects for unit tests.

Author: Cristian Perera
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return {
        'Time': 150000,
        'Amount': 100.50,
        'V1': -1.5, 'V2': 0.8, 'V3': 1.2, 'V4': -0.5,
        'V5': 0.3, 'V6': -0.7, 'V7': 0.9, 'V8': 0.2,
        'V9': -0.4, 'V10': 1.1, 'V11': -0.8, 'V12': 0.6,
        'V13': -1.2, 'V14': -5.5, 'V15': 0.4, 'V16': -0.9,
        'V17': 0.3, 'V18': 0.7, 'V19': -0.3, 'V20': 0.5,
        'V21': -0.6, 'V22': 0.8, 'V23': -0.2, 'V24': 0.4,
        'V25': 0.1, 'V26': -0.5, 'V27': 0.2, 'V28': 0.3
    }


@pytest.fixture
def fraud_transaction():
    """Create a fraudulent transaction with high V14 anomaly."""
    return {
        'Time': 160000,
        'Amount': 1.00,
        'V1': -2.1, 'V2': 1.5, 'V3': -0.8, 'V4': 2.3,
        'V5': -1.2, 'V6': 1.8, 'V7': -1.5, 'V8': 0.9,
        'V9': -2.3, 'V10': 1.6, 'V11': 2.1, 'V12': -1.8,
        'V13': 0.7, 'V14': -10.5, 'V15': -1.2, 'V16': 2.5,
        'V17': -0.8, 'V18': 1.3, 'V19': -2.1, 'V20': 0.8,
        'V21': -1.5, 'V22': 1.7, 'V23': -0.9, 'V24': 1.2,
        'V25': -0.6, 'V26': 1.4, 'V27': -0.5, 'V28': 0.7
    }


@pytest.fixture
def safe_transaction():
    """Create a safe transaction with normal features."""
    return {
        'Time': 170000,
        'Amount': 50.00,
        'V1': 0.2, 'V2': -0.1, 'V3': 0.3, 'V4': -0.2,
        'V5': 0.1, 'V6': 0.0, 'V7': -0.1, 'V8': 0.2,
        'V9': 0.0, 'V10': -0.3, 'V11': 0.1, 'V12': -0.2,
        'V13': 0.2, 'V14': 0.5, 'V15': 0.1, 'V16': -0.1,
        'V17': 0.0, 'V18': 0.2, 'V19': -0.1, 'V20': 0.1,
        'V21': 0.0, 'V22': -0.2, 'V23': 0.1, 'V24': 0.0,
        'V25': 0.1, 'V26': -0.1, 'V27': 0.0, 'V28': 0.1
    }


@pytest.fixture
def sample_dataframe():
    """Create a small DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.uniform(0, 500, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate for testing
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_xgboost_model():
    """Create a mock XGBoost model for testing without training."""
    mock_model = MagicMock()
    
    # Mock predict_proba to return realistic fraud scores
    def mock_predict_proba(X):
        n_samples = len(X) if hasattr(X, '__len__') else 1
        # Return low fraud probabilities for most cases
        probs = np.random.uniform(0.001, 0.05, n_samples)
        # Stack as [safe_prob, fraud_prob]
        return np.column_stack([1 - probs, probs])
    
    mock_model.predict_proba = Mock(side_effect=mock_predict_proba)
    mock_model.predict = Mock(return_value=np.array([0]))
    
    return mock_model


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing LLM calls."""
    mock_client = MagicMock()
    
    # Mock the response structure
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        "The fraud case shows suspicious patterns with extreme V14 values "
        "indicating potential card-not-present fraud. The model detected "
        "this anomaly with high confidence."
    )
    
    mock_client.chat.completions.create = Mock(return_value=mock_response)
    
    return mock_client


@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary path for model saving/loading tests."""
    return tmp_path / "test_model.pkl"


@pytest.fixture
def mock_csv_path(tmp_path, sample_dataframe):
    """Create a temporary CSV file for data loading tests."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)
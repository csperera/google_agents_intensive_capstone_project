"""
Unit Tests for FraudAgent
==========================
Tests fraud scoring tool, LLM integration, and failover logic.

Authors: Ash Dehghan Ph.D and Cristian Perera
Date: November 2025
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fraud_agent import FraudAgent, create_agent_from_model_path


class TestFraudAgent:
    """Test suite for FraudAgent class."""
    
    def test_agent_initialization(self, mock_xgboost_model):
        """Test agent initializes with model and API client."""
        api_key = "test-key-123"
        agent = FraudAgent(mock_xgboost_model, api_key)
        
        assert agent.model is not None
        assert agent.client is not None
        assert len(agent.FREE_MODELS) == 4
    
    def test_xgboost_fraud_score_low_risk(self, mock_xgboost_model, safe_transaction):
        """Test fraud scoring for safe transaction."""
        # Mock low fraud probability
        mock_xgboost_model.predict_proba = Mock(
            return_value=np.array([[0.98, 0.02]])  # 2% fraud prob
        )
        
        agent = FraudAgent(mock_xgboost_model, "test-key")
        result = agent.xgboost_fraud_score(safe_transaction)
        
        assert "0.0200" in result  # Check probability
        assert "LOW — SAFE" in result
        assert "Amount:" in result
        assert "Time:" in result
    
    def test_xgboost_fraud_score_high_risk(self, mock_xgboost_model, fraud_transaction):
        """Test fraud scoring for fraudulent transaction."""
        # Mock high fraud probability
        mock_xgboost_model.predict_proba = Mock(
            return_value=np.array([[0.01, 0.99]])  # 99% fraud prob
        )
        
        agent = FraudAgent(mock_xgboost_model, "test-key")
        result = agent.xgboost_fraud_score(fraud_transaction)
        
        assert "0.9900" in result
        assert "EXTREMELY HIGH" in result
        assert "BLOCK IMMEDIATELY" in result
    
    def test_xgboost_fraud_score_medium_risk(self, mock_xgboost_model, sample_transaction):
        """Test fraud scoring for medium risk transaction."""
        # Mock medium fraud probability
        mock_xgboost_model.predict_proba = Mock(
            return_value=np.array([[0.55, 0.45]])  # 45% fraud prob
        )
        
        agent = FraudAgent(mock_xgboost_model, "test-key")
        result = agent.xgboost_fraud_score(sample_transaction)
        
        assert "0.4500" in result
        assert "MEDIUM" in result
        assert "MONITOR CLOSELY" in result
    
    def test_analyze_transactions_success(
        self, 
        mock_xgboost_model, 
        mock_openai_client,
        fraud_transaction,
        safe_transaction
    ):
        """Test successful transaction analysis with LLM."""
        agent = FraudAgent(mock_xgboost_model, "test-key")
        agent.client = mock_openai_client
        
        # Convert dicts to Series
        fraud_series = pd.Series(fraud_transaction)
        safe_series = pd.Series(safe_transaction)
        
        result = agent.analyze_transactions(fraud_series, safe_series)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        mock_openai_client.chat.completions.create.assert_called()
    
    def test_analyze_transactions_failover(
        self,
        mock_xgboost_model,
        fraud_transaction,
        safe_transaction
    ):
        """Test LLM failover when primary model fails."""
        agent = FraudAgent(mock_xgboost_model, "test-key")
        
        # Mock client that fails for first model, succeeds for second
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Analysis from backup model"
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Primary model unavailable")
            return mock_response
        
        agent.client.chat.completions.create = Mock(side_effect=side_effect)
        
        fraud_series = pd.Series(fraud_transaction)
        safe_series = pd.Series(safe_transaction)
        
        result = agent.analyze_transactions(
            fraud_series, 
            safe_series,
            max_retries=2
        )
        
        assert result is not None
        assert "Analysis from backup model" in result
        assert call_count == 2  # Failed once, succeeded on second try
    
    def test_analyze_transactions_all_models_fail(
        self,
        mock_xgboost_model,
        fraud_transaction,
        safe_transaction,
        capsys
    ):
        """Test behavior when all LLM models fail."""
        agent = FraudAgent(mock_xgboost_model, "test-key")
        
        # Mock client that always fails
        agent.client.chat.completions.create = Mock(
            side_effect=Exception("All models down")
        )
        
        fraud_series = pd.Series(fraud_transaction)
        safe_series = pd.Series(safe_transaction)
        
        result = agent.analyze_transactions(
            fraud_series,
            safe_series,
            max_retries=3
        )
        
        assert result is None
        captured = capsys.readouterr()
        assert "All free models failed" in captured.out
    
    def test_create_analysis_prompt(self, mock_xgboost_model):
        """Test prompt generation for LLM."""
        agent = FraudAgent(mock_xgboost_model, "test-key")
        
        fraud_tx = pd.Series({'Amount': 100.0, 'V14': -10.5, 'Time': 150000})
        safe_tx = pd.Series({'Amount': 50.0, 'V14': 0.5, 'Time': 160000})
        
        prompt = agent._create_analysis_prompt(fraud_tx, 0.98, safe_tx, 0.02)
        
        assert "fraud detection analyst" in prompt.lower()
        assert "0.9800" in prompt
        assert "0.0200" in prompt
        assert "100.00" in prompt
        assert "50.00" in prompt
    
    def test_format_response(self, mock_xgboost_model):
        """Test text wrapping for LLM responses."""
        agent = FraudAgent(mock_xgboost_model, "test-key")
        
        long_text = "This is a very long sentence " * 20
        wrapped = agent._format_response(long_text, width=80)
        
        lines = wrapped.split('\n')
        for line in lines:
            assert len(line) <= 85  # Allow some buffer for word boundaries


class TestCreateAgentFromModelPath:
    """Test the convenience function for creating agents."""
    
    @patch('fraud_agent.joblib.load')
    @patch('fraud_agent.os.getenv')
    def test_create_agent_with_api_key(self, mock_getenv, mock_load, mock_xgboost_model):
        """Test agent creation with explicit API key."""
        mock_load.return_value = mock_xgboost_model
        
        agent = create_agent_from_model_path(
            model_path="models/test.pkl",
            api_key="explicit-key"
        )
        
        assert agent is not None
        assert isinstance(agent, FraudAgent)
        mock_load.assert_called_once_with("models/test.pkl")
    
    @patch('fraud_agent.joblib.load')
    @patch('fraud_agent.os.getenv')
    def test_create_agent_from_env(self, mock_getenv, mock_load, mock_xgboost_model):
        """Test agent creation using environment variable."""
        mock_getenv.return_value = "env-api-key"
        mock_load.return_value = mock_xgboost_model
        
        agent = create_agent_from_model_path(model_path="models/test.pkl")
        
        assert agent is not None
        mock_getenv.assert_called_once_with("OPENROUTER_API_KEY")
    
    @patch('fraud_agent.joblib.load')
    @patch('fraud_agent.os.getenv')
    def test_create_agent_missing_key(self, mock_getenv, mock_load, mock_xgboost_model):
        """Test that missing API key raises error."""
        mock_getenv.return_value = None
        mock_load.return_value = mock_xgboost_model
        
        with pytest.raises(ValueError, match="API key required"):
            create_agent_from_model_path(model_path="models/test.pkl")


# Add numpy import that was missing
import numpy as np
"""
AI-Powered Fraud Detection Agent
=================================
Combines XGBoost predictions with LLM explainability for credit card fraud analysis.

Author: Cristian Perera
Date: November 2025
"""

import pandas as pd
import textwrap
from openai import OpenAI
from typing import Dict, Optional
from pathlib import Path
import sys


class FraudAgent:
    """
    AI agent that provides human-interpretable fraud analysis by combining
    ML predictions with LLM-generated explanations.
    """
    
    # Free-tier LLM models (in priority order)
    FREE_MODELS = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2-7b-instruct:free"
    ]
    
    def __init__(self, model, api_key: str):
        """
        Initialize the fraud detection agent.
        
        Args:
            model: Trained XGBoost model instance
            api_key: OpenRouter API key
        """
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
    def xgboost_fraud_score(self, transaction: Dict) -> str:
        """
        Generate fraud risk assessment for a single transaction.
        
        Args:
            transaction: Dictionary of transaction features
            
        Returns:
            Formatted string with fraud probability and risk level
        """
        row = pd.DataFrame([transaction])
        prob = self.model.predict_proba(row)[0][1]
        
        # Risk classification
        if prob > 0.95:
            risk = "EXTREMELY HIGH — BLOCK IMMEDIATELY"
        elif prob > 0.70:
            risk = "HIGH — ALERT & MANUAL REVIEW"
        elif prob > 0.30:
            risk = "MEDIUM — MONITOR CLOSELY"
        else:
            risk = "LOW — SAFE"
        
        return f"""
XGBoost Fraud Probability: {prob:.4f}
Risk Level: {risk}
Confidence: {(prob if prob > 0.5 else 1-prob):.1%}

Top Features:
→ Amount: ${transaction.get('Amount', 0):.2f}
→ Time: {transaction.get('Time', 0)//3600:.0f}h
→ V14: {transaction.get('V14', 0):.2f} | V17: {transaction.get('V17', 0):.2f}
        """.strip()
    
    def analyze_transactions(
        self, 
        fraud_transaction: pd.Series,
        safe_transaction: pd.Series,
        temperature: float = 0.2,
        max_retries: int = 4
    ) -> Optional[str]:
        """
        Generate AI-powered comparative analysis of fraud vs safe transactions.
        
        Args:
            fraud_transaction: Feature vector for a fraudulent transaction
            safe_transaction: Feature vector for a safe transaction
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Number of models to try before giving up
            
        Returns:
            Human-readable analysis text, or None if all models fail
        """
        # Calculate fraud scores
        fraud_score = self.model.predict_proba(
            fraud_transaction.values.reshape(1, -1)
        )[0][1]
        safe_score = self.model.predict_proba(
            safe_transaction.values.reshape(1, -1)
        )[0][1]
        
        # Print transaction details
        self._print_transaction_details(
            fraud_transaction, fraud_score,
            safe_transaction, safe_score
        )
        
        # Create prompt for LLM
        prompt = self._create_analysis_prompt(
            fraud_transaction, fraud_score,
            safe_transaction, safe_score
        )
        
        # Try models with failover
        for model_name in self.FREE_MODELS[:max_retries]:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                print(f"\n✓ Successfully used model: {model_name}")
                return self._format_response(response.choices[0].message.content)
                
            except Exception as e:
                print(f"✗ {model_name} failed, trying next...")
                continue
        
        print("✗ All free models failed. Check OpenRouter status or use a paid model.")
        return None
    
    def _print_transaction_details(
        self,
        fraud_tx: pd.Series,
        fraud_score: float,
        safe_tx: pd.Series,
        safe_score: float
    ) -> None:
        """Print formatted transaction comparison."""
        print("=" * 80)
        print("TRANSACTION DETAILS")
        print("=" * 80)
        print(f"\n🚨 FRAUD CASE (Score: {fraud_score:.4f})")
        print(f"   Amount: ${fraud_tx['Amount']:.2f}")
        print(f"   Time: {fraud_tx['Time']:.0f}s ({fraud_tx['Time']//3600:.0f}h)")
        print(f"   V14: {fraud_tx['V14']:.2f} | V17: {fraud_tx['V17']:.2f}")
        
        print(f"\n✅ SAFE CASE (Score: {safe_score:.4f})")
        print(f"   Amount: ${safe_tx['Amount']:.2f}")
        print(f"   Time: {safe_tx['Time']:.0f}s ({safe_tx['Time']//3600:.0f}h)")
        print(f"   V14: {safe_tx['V14']:.2f} | V17: {safe_tx['V17']:.2f}")
        print("=" * 80)
    
    def _create_analysis_prompt(
        self,
        fraud_tx: pd.Series,
        fraud_score: float,
        safe_tx: pd.Series,
        safe_score: float
    ) -> str:
        """Generate prompt for LLM analysis."""
        return f"""
You are an elite fraud detection analyst.

FRAUD CASE (score {fraud_score:.4f}):
Amount ${fraud_tx['Amount']:.2f}, V14 {fraud_tx['V14']:.2f}

SAFE CASE (score {safe_score:.4f}):
Amount ${safe_tx['Amount']:.2f}, V14 {safe_tx['V14']:.2f}

Explain in plain English why the fraud case is suspicious and how the model caught it.
        """.strip()
    
    def _format_response(self, text: str, width: int = 80) -> str:
        """Format LLM response with word wrapping."""
        return textwrap.fill(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False
        )


def create_agent_from_model_path(
    model_path: str = "models/xgboost_fraud_model.pkl",
    api_key: Optional[str] = None
) -> FraudAgent:
    """
    Convenience function to load model and create agent.
    
    Args:
        model_path: Path to saved XGBoost model
        api_key: OpenRouter API key (reads from env if not provided)
        
    Returns:
        Initialized FraudAgent instance
    """
    import joblib
    import os
    
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
    
    model = joblib.load(model_path)
    return FraudAgent(model, api_key)


if __name__ == "__main__":
    """
    Demo: Run agent analysis on random fraud vs safe cases.
    """
    import joblib
    
    # Load model
    model = joblib.load("models/xgboost_fraud_model.pkl")
    
    # Load test data
    df = pd.read_csv("data/creditcard.csv")
    test = df.iloc[227845:]
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]
    
    # Create agent with your API key
    api_key = "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"
    agent = FraudAgent(model, api_key)
    
    # Select random cases
    fraud_case = X_test[y_test == 1].sample(n=1).iloc[0]
    safe_case = X_test[y_test == 0].sample(n=1).iloc[0]
    
    # Run analysis
    print("\n🚀 Running AI Fraud Analysis...\n")
    analysis = agent.analyze_transactions(fraud_case, safe_case)
    
    if analysis:
        print("\n" + "=" * 80)
        print("AI AGENT ANALYSIS")
        print("=" * 80)
        print(analysis)
        print("=" * 80)
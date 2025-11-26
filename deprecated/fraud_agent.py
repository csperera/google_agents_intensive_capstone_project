"""
AI-Powered Fraud Detection Agent
=================================
Combines XGBoost predictions with LLM explainability for credit card fraud analysis.

Authors: Cristian Perera & Ash Dehghan Ph.D
Date: November 2025
"""

import pandas as pd
import textwrap
from google import genai
from typing import Dict, Optional
from pathlib import Path
import sys
import os
import joblib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FraudAgent:
    """
    AI agent that provides human-interpretable fraud analysis by combining
    ML predictions with LLM-generated explanations.
    """
    
    # Google AI Studio models (in priority order for failover)
    # Note: Model names must include the 'models/' prefix
    FREE_MODELS = [
        'models/gemini-2.5-flash',                     # Fast, reliable (confirmed available)
        'models/gemini-2.5-pro-preview-06-05',         # Latest preview (confirmed available)
        'models/gemini-2.5-pro-preview-05-06',         # Previous preview
        'models/gemini-2.5-pro-preview-03-25',         # Older preview
        'models/gemini-1.5-flash',                     # Stable fallback
        'models/gemini-1.5-pro'                        # More capable fallback
    ]
    
    def __init__(self, model, api_key: str):
        """
        Initialize the fraud detection agent.
        
        Args:
            model: Trained XGBoost model instance
            api_key: Google AI Studio API key
        """
        self.model = model
        self.client = genai.Client(api_key=api_key)
        
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
        max_retries: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate AI-powered comparative analysis of fraud vs safe transactions.
        
        Args:
            fraud_transaction: Feature vector for a fraudulent transaction
            safe_transaction: Feature vector for a safe transaction
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Number of models to try before giving up (default: all available models)
            
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
        
        # Try models with failover (use all models if max_retries not specified)
        models_to_try = self.FREE_MODELS[:max_retries] if max_retries else self.FREE_MODELS
        
        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                analysis = response.text
                print(f"\n✓ Successfully used model: {model_name}")
                return self._format_response(analysis)
                        
            except Exception as e:
                print(f"✗ {model_name} failed, trying next...")
                continue
        
        print("✗ All Google models failed. Please check your Google AI API connection.")
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
        print(f"   Amount: ${fraud_tx['Amount']:.2f}")
        print(f"   Time: {fraud_tx['Time']:.0f}s ({fraud_tx['Time']//3600:.0f}h)")
        print(f"   V14: {fraud_tx['V14']:.2f} | V17: {fraud_tx['V17']:.2f}")
        
        print(f"\n✅ SAFE CASE (Score: {safe_score:.4f})")
        print(f"   Amount: ${safe_tx['Amount']:.2f}")
        print(f"   Time: {safe_tx['Time']:.0f}s ({safe_tx['Time']//3600:.0f}h)")
        print(f"   V14: {safe_tx['V14']:.2f} | V17: {safe_tx['V17']:.2f}")
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

Provide a clear, structured analysis using bullet points. Format your response as:

FRAUD CASE ANALYSIS:
• Key indicator 1: [explanation]
• Key indicator 2: [explanation]
• How model detected it: [explanation]

SAFE CASE ANALYSIS:
• Why it's safe: [explanation]
• Key differences from fraud case: [explanation]

SUMMARY:
• [Brief conclusion]

Use bullet points (•) for each point. Keep explanations concise and focused.
        """.strip()
    
    def _format_response(self, text: str, width: int = 80) -> str:
        """Format LLM response preserving structure and line breaks."""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # If line is already a bullet point or starts with special chars, preserve it
            if line.startswith(('•', '-', '*', '#', '**')) or line.startswith(('1.', '2.', '3.', '4.', '5.')):
                # Wrap long bullet points but preserve the bullet
                if len(line) > width:
                    # Find the bullet/prefix
                    prefix = ''
                    content = line
                    for prefix_char in ['•', '-', '*']:
                        if line.startswith(prefix_char):
                            prefix = prefix_char + ' '
                            content = line[len(prefix_char):].strip()
                            break
                    
                    # Wrap the content
                    wrapped = textwrap.fill(
                        content,
                        width=width - len(prefix),
                        break_long_words=False,
                        break_on_hyphens=False
                    )
                    # Add prefix to first line, indent subsequent lines
                    wrapped_lines = wrapped.split('\n')
                    formatted_lines.append(prefix + wrapped_lines[0])
                    for wrapped_line in wrapped_lines[1:]:
                        formatted_lines.append(' ' * len(prefix) + wrapped_line)
                else:
                    formatted_lines.append(line)
            else:
                # Regular text - wrap it
                wrapped = textwrap.fill(
                    line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False
                )
                formatted_lines.extend(wrapped.split('\n'))
        
        return '\n'.join(formatted_lines)


def create_agent_from_model_path(
    model_path: Path, # Changed type hint to Path
    api_key: Optional[str] = None
) -> FraudAgent:
    """
    Convenience function to load model and create agent.
    
    Args:
        model_path: Path to saved XGBoost model
        api_key: Google AI Studio API key (reads from env if not provided)
        
    Returns:
        Initialized FraudAgent instance
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key required. Set GOOGLE_AI_API_KEY env var or pass api_key parameter.\n"
                "Get your API key from: https://aistudio.google.com/"
            )
    
    # model_path is now passed as a Path object
    model = joblib.load(model_path)
    return FraudAgent(model, api_key)


if __name__ == "__main__":
    """
    Demo: Run agent analysis on random fraud vs safe cases.
    """
    # Define script and data paths robustly using pathlib
    SCRIPT_DIR = Path(__file__).resolve().parent
    MODEL_PATH = SCRIPT_DIR.parent / "models" / "xgboost_fraud_model.pkl"
    DATA_PATH = SCRIPT_DIR.parent / "data" / "creditcard.csv"
    
    # Load API key from environment
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_AI_API_KEY not found in environment variables.\n"
            "Set it with: export GOOGLE_AI_API_KEY='your-key-here'\n"
            "Or add it to your .env file\n"
            "Get your API key from: https://aistudio.google.com/"
        )
    
    # Load model
    model = joblib.load(MODEL_PATH)
    
    # Load test data
    df = pd.read_csv(DATA_PATH)
    test = df.iloc[227845:]
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]
    
    # Create agent
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
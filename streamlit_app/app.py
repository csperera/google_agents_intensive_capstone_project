import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from openai import OpenAI

# Page config
st.set_page_config(
    page_title="Fraud Detection Simulator",
    page_icon="💳",
    layout="wide"
)

# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load("xgboost_fraud_model.pkl")

# Initialize OpenRouter client
@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )

try:
    model = load_model()
    client = get_client()
except:
    st.error("⚠️ Model file not found. Make sure 'xgboost_fraud_model.pkl' is in the same directory.")
    st.stop()

# Header
st.title("💳 Real-Time Fraud Detection Simulator")
st.markdown("Enter transaction details below to get instant AI-powered fraud analysis")
st.divider()

# Sidebar for input
with st.sidebar:
    st.header("📝 Transaction Input")
    
    # Sliders
    amount = st.slider(
        "Transaction Amount ($)",
        min_value=0.0,
        max_value=500.0,
        value=125.5,
        step=0.5,
        help="Amount in dollars"
    )
    
    time_hours = st.slider(
        "Time (hours since first transaction)",
        min_value=0,
        max_value=48,
        value=12,
        help="Time elapsed in hours"
    )
    
    v14 = st.slider(
        "V14 (PCA Feature)",
        min_value=-10.0,
        max_value=10.0,
        value=-2.5,
        step=0.1,
        help="Principal component 14"
    )
    
    v17 = st.slider(
        "V17 (PCA Feature)",
        min_value=-10.0,
        max_value=10.0,
        value=-1.8,
        step=0.1,
        help="Principal component 17"
    )
    
    st.divider()
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Random", use_container_width=True):
            # Generate random transaction
            is_fraud = np.random.random() > 0.5
            if is_fraud:
                st.session_state.amount = np.random.uniform(0.5, 50.0)
                st.session_state.v14 = np.random.uniform(-10, -2)
                st.session_state.v17 = np.random.uniform(-10, -2)
            else:
                st.session_state.amount = np.random.uniform(20.0, 300.0)
                st.session_state.v14 = np.random.uniform(-2, 2)
                st.session_state.v17 = np.random.uniform(-2, 2)
            st.session_state.time_hours = np.random.randint(0, 48)
            st.rerun()
    
    with col2:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Use session state values if they exist
if 'amount' in st.session_state:
    amount = st.session_state.amount
if 'time_hours' in st.session_state:
    time_hours = st.session_state.time_hours
if 'v14' in st.session_state:
    v14 = st.session_state.v14
if 'v17' in st.session_state:
    v17 = st.session_state.v17

# Create transaction dictionary with all required features
# CRITICAL: Order must match training data (Amount comes LAST)
transaction = {
    'Time': time_hours * 3600,
    'V1': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'V5': 0,
    'V6': 0, 'V7': 0, 'V8': 0, 'V9': 0, 'V10': 0,
    'V11': 0, 'V12': 0, 'V13': 0, 'V14': v14, 'V15': 0,
    'V16': 0, 'V17': v17, 'V18': 0, 'V19': 0, 'V20': 0,
    'V21': 0, 'V22': 0, 'V23': 0, 'V24': 0, 'V25': 0,
    'V26': 0, 'V27': 0, 'V28': 0,
    'Amount': amount  # Amount must be LAST
}

# Make prediction
df_transaction = pd.DataFrame([transaction])
fraud_prob = model.predict_proba(df_transaction)[0][1]

# Determine risk level
if fraud_prob > 0.95:
    risk_level = "🔴 EXTREMELY HIGH - BLOCK IMMEDIATELY"
    risk_color = "red"
elif fraud_prob > 0.70:
    risk_level = "🟠 HIGH - ALERT & MANUAL REVIEW"
    risk_color = "orange"
elif fraud_prob > 0.30:
    risk_level = "🟡 MEDIUM - MONITOR CLOSELY"
    risk_color = "yellow"
else:
    risk_level = "🟢 LOW - SAFE"
    risk_color = "green"

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Fraud Risk Assessment")
    
    # Metric display
    st.metric(
        label="Fraud Probability",
        value=f"{fraud_prob:.1%}",
        delta=None
    )
    
    # Progress bar colored by risk
    st.progress(float(fraud_prob))
    
    st.markdown(f"**Risk Level:** {risk_level}")
    st.markdown(f"**Confidence:** {max(fraud_prob, 1-fraud_prob):.1%}")
    
    st.divider()
    
    # Feature impact (simplified)
    st.subheader("📊 Feature Impact")
    
    # Calculate relative impacts (simplified for demo)
    v14_impact = min(abs(v14) / 10, 1.0)
    v17_impact = min(abs(v17) / 10, 1.0)
    amount_impact = min(amount / 500, 0.5)
    time_impact = 0.1
    
    st.progress(v14_impact, text=f"V14 Value: {v14_impact:.0%} impact")
    st.progress(v17_impact, text=f"V17 Value: {v17_impact:.0%} impact")
    st.progress(amount_impact, text=f"Amount: {amount_impact:.0%} impact")
    st.progress(time_impact, text=f"Time: {time_impact:.0%} impact")

with col2:
    st.subheader("🤖 AI Fraud Analyst")
    
    if analyze_btn or 'last_analysis' not in st.session_state:
        with st.spinner("Analyzing transaction..."):
            # Create prompt for AI
            prompt = f"""You are an elite fraud detection analyst.

Analyze this transaction:
- Amount: ${amount:.2f}
- Time: {time_hours}h since first transaction
- V14: {v14:.2f}
- V17: {v17:.2f}
- Fraud Probability: {fraud_prob:.1%}

Provide a concise explanation (3-4 sentences) of:
1. Why this transaction received this fraud score
2. What patterns the model detected
3. Your recommendation (approve/review/block)

Be specific about the V14 and V17 values and their significance."""

            # Try models
            free_models = [
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemma-2-9b-it:free",
                "mistralai/mistral-7b-instruct:free",
            ]
            
            analysis = None
            for model_name in free_models:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    )
                    analysis = response.choices[0].message.content
                    break
                except:
                    continue
            
            if analysis:
                st.session_state.last_analysis = analysis
            else:
                st.session_state.last_analysis = "AI analysis unavailable. Please check your API connection."
    
    # Display analysis
    if 'last_analysis' in st.session_state:
        st.markdown(st.session_state.last_analysis)

# Footer
st.divider()
st.caption("🎓 Capstone Project: AI-Powered Credit Card Fraud Detection | XGBoost Model (AUC: 0.9886)")

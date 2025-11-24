import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Page config - full width, no sidebar
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none}
        .block-container {padding-top: 2rem; padding-left: 1rem; padding-right: 1rem;}
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_resources():
    # Try different possible model paths
    possible_model_paths = [
        "../models/xgboost_fraud_model.pkl",
        "models/xgboost_fraud_model.pkl",
        "xgboost_fraud_model.pkl"
    ]
    
    model = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
    
    if model is None:
        raise FileNotFoundError(
            "Model file not found. Please run 'python src/model.py' to train the model first."
        )
    
    # Load test data (try relative paths)
    possible_data_paths = [
        "../data/creditcard.csv",
        "data/creditcard.csv",
        r"C:\Users\chris\google_agents_intensive_capstone_project\data\creditcard.csv"
    ]
    
    df = None
    for path in possible_data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Credit card dataset not found. Please check data/creditcard.csv")
    
    train = df.iloc[:227845]
    test = df.iloc[227845:]
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]
    return model, X_test, y_test

@st.cache_resource
def get_ai_client():
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_AI_API_KEY not found in .env file.\n"
            "Get your API key from: https://aistudio.google.com/"
        )
    return genai.Client(api_key=api_key)

try:
    model, X_test, y_test = load_resources()
    client = get_ai_client()
except Exception as e:
    st.error(f"⚠️ Error loading resources: {e}")
    st.stop()

# Initialize session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'fraud_detected' not in st.session_state:
    st.session_state.fraud_detected = False
if 'fraud_data' not in st.session_state:
    st.session_state.fraud_data = None
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'fraud_count' not in st.session_state:
    st.session_state.fraud_count = 0

# Header with ticker in top right
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🚨 Real-Time Credit Card Fraud Detection Agent")
with header_col2:
    # Ticker placeholder for top right
    ticker_placeholder = st.empty()
    if 'ticker_display' not in st.session_state:
        st.session_state.ticker_display = ""

st.markdown("---")

# Main button logic
if not st.session_state.monitoring and not st.session_state.fraud_detected:
    # Show big START button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚨 BEGIN FRAUD DETECTION", type="primary", use_container_width=True):
            st.session_state.monitoring = True
            st.session_state.current_index = 0
            st.session_state.total_processed = 0
            st.session_state.fraud_count = 0
            st.rerun()

elif st.session_state.fraud_detected:
    # Show fraud alert and analysis
    st.error("🚨 FRAUD DETECTED - MONITORING PAUSED")
    
    fraud_tx = st.session_state.fraud_data['transaction']
    fraud_prob = st.session_state.fraud_data['probability']
    
    # Display fraud details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Card Number", f"**** **** **** {np.random.randint(1000, 9999)}")
    with col2:
        st.metric("Amount", f"${fraud_tx['Amount']:.2f}")
    with col3:
        st.metric("Fraud Score", f"{fraud_prob:.1%}")
    
    # Determine risk level
    if fraud_prob > 0.95:
        risk_level = "🔴 EXTREMELY HIGH - BLOCK IMMEDIATELY"
    elif fraud_prob > 0.70:
        risk_level = "🟠 HIGH - ALERT & MANUAL REVIEW"
    else:
        risk_level = "🟡 MEDIUM - MONITOR CLOSELY"
    
    st.markdown(f"**Risk Level:** {risk_level}")
    st.markdown("---")
    
    # AI Analysis
    st.subheader("🤖 AI Fraud Agent Report")
    
    if 'ai_analysis' not in st.session_state:
        with st.spinner("🔍 AI analyzing transaction patterns..."):
            # Create prompt
            prompt = f"""You are a machine learning model explainer analyzing fraud detection outputs.

A fraud detection model flagged this transaction as fraudulent:
- Transaction Amount: ${fraud_tx['Amount']:.2f}
- Time Since First Transaction: {fraud_tx['Time']//3600:.0f} hours
- Feature V14 (PCA component): {fraud_tx['V14']:.2f}
- Feature V17 (PCA component): {fraud_tx['V17']:.2f}
- Model's Fraud Probability Score: {fraud_prob:.1%}

Provide a detailed fraud analysis (4-5 sentences) explaining:
1. Why this transaction is highly suspicious
2. What specific patterns triggered the high fraud score
3. What the V14 and V17 values indicate about fraudulent behavior
4. What action should be taken

This is for an educational Google Capstone project demonstrating AI-powered fraud detection."""

            # Try Google models (must include 'models/' prefix)
            google_models = [
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro-preview-06-05',
                'models/gemini-2.5-pro-preview-05-06',
                'models/gemini-2.5-pro-preview-03-25',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro'
            ]

            analysis = None
            for model_name in google_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    analysis = response.text
                    break
                except:
                    continue
            
            if analysis:
                st.session_state.ai_analysis = analysis
            else:
                st.session_state.ai_analysis = "AI analysis unavailable. Manual review required."
    
    # Display analysis
    st.markdown(st.session_state.ai_analysis)
    st.markdown("---")
    
    # Resume button (with unique key to prevent duplication)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶ RESUME MONITORING", type="primary", use_container_width=True, key="resume_monitoring_btn"):
            st.session_state.fraud_detected = False
            st.session_state.monitoring = True
            st.session_state.fraud_data = None
            if 'ai_analysis' in st.session_state:
                del st.session_state.ai_analysis
            st.rerun()

elif st.session_state.monitoring:
    # Active monitoring - process transactions
    st.success("✅ MONITORING ACTIVE - Processing transactions...")
    
    # Stats display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transactions Processed", st.session_state.total_processed)
    with col2:
        st.metric("Frauds Detected", st.session_state.fraud_count)
    with col3:
        st.metric("Processing Speed", "20 tx/sec")
    
    # Process transactions
    for i in range(10):  # Process 10 at a time
        if st.session_state.current_index >= len(X_test):
            st.session_state.current_index = 0  # Loop back
        
        # Get transaction
        tx = X_test.iloc[st.session_state.current_index]
        actual_label = y_test.iloc[st.session_state.current_index]
        
        # Predict
        prob = model.predict_proba(tx.values.reshape(1, -1))[0][1]
        
        # Update ticker in top right with huge font
        card_num = f"**** {np.random.randint(1000, 9999)}"
        status = "✅" if prob < 0.95 else "🚨"
        ticker_html = f"""
        <div style="text-align: right; font-size: 3rem; font-weight: bold; padding: 10px;">
            💳 {card_num}<br>
            ${tx['Amount']:.2f} {status}
        </div>
        """
        ticker_placeholder.markdown(ticker_html, unsafe_allow_html=True)
        
        st.session_state.total_processed += 1
        st.session_state.current_index += 1
        
        # Check for fraud
        if prob > 0.95:
            # FRAUD DETECTED!
            st.session_state.fraud_detected = True
            st.session_state.monitoring = False
            st.session_state.fraud_count += 1
            st.session_state.fraud_data = {
                'transaction': tx.to_dict(),
                'probability': prob
            }
            
            # Play alarm sound
            st.markdown("""
                <audio autoplay>
                    <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTGH0fPTgjMGHm7A7+OZSA0PVa3n7qlXEwpBmuL0wm0gBjKM0vPVgzQGH2q/7uGaTRANUqzk7qVUEgw+lODzvWsgBzWS1PPZgjIGH2q/7uCZThENUark7qRUEQw9lODztmwiBjiM0vPXgzIFIG7A7uCZTQ4NUark7qRUEgtAmeD0t2wiBjiR1fPYgjQGIG6/7+GaTQ8MUqvm7qVUEwxBl+D0uG4iB" type="audio/wav">
                </audio>
            """, unsafe_allow_html=True)
            
            time.sleep(0.2)  # Brief pause for fraud alert
            st.rerun()
        
        time.sleep(0.05)  # 20 tx/sec (1/20 = 0.05 seconds per transaction)
    
    # Continue monitoring
    st.rerun()

# Footer
st.markdown("---")
st.caption("🎓 Google Capstone Project: AI-Powered Credit Card Fraud Detection | XGBoost Model (AUC: 0.9886)")
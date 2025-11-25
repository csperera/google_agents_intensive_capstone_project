import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig 
import sys
from pathlib import Path 

# --- ROBUST FIX FOR MODULE NOT FOUND ERROR ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / 'src')) 
# --- END ROBUST FIX ---

# --- Import the Multi-Agent Classes ---
from multi_agent_fraud import PredictionAgent, TriageAgent, ExplanationAgent

# Load environment variables
load_dotenv()

# Page config (Removed 'collapsed' state to show sidebar)
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="auto" # Changed to 'auto' to enable sidebar use
)

# Optional: Hide sidebar collapse button if needed, but we want the sidebar visible
st.markdown("""
    <style>
        /* Removed [data-testid="collapsedControl"] {display: none} to keep sidebar open */
        .block-container {padding-top: 2rem; padding-left: 1rem; padding-right: 1rem;}
    </style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    # Try different possible model paths
    possible_model_paths = [
        "xgboost_fraud_model.pkl",
        "models/xgboost_fraud_model.pkl",
        "../models/xgboost_fraud_model.pkl"
    ]
    
    model = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
    
    if model is None:
        raise FileNotFoundError("Model file not found. Ensure xgboost_fraud_model.pkl exists.")
    
    # Load test data (try relative paths)
    possible_data_paths = [
        "data/creditcard.csv",
        "creditcard.csv",
        "../data/creditcard.csv"
    ]
    
    df = None
    for path in possible_data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Credit card dataset not found.")
    
    train = df.iloc[:227845]
    test = df.iloc[227845:]
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]
    
    # Return core components
    return model, X_test, y_test

# --- Function to Instantiate Agents (FIXED CACHING) ---
@st.cache_resource
def get_agents(_model): # ADDED UNDERSCORE to ignore hashing
    """Instantiates the multi-agent system components."""
    
    # Use the ignored model argument
    prediction_agent = PredictionAgent(_model)
    triage_agent = TriageAgent()
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_AI_API_KEY not found. ExplanationAgent will be disabled.")
        explanation_agent = None 
    else:
        explanation_agent = ExplanationAgent(api_key=api_key)
        
    return prediction_agent, triage_agent, explanation_agent
# --- END FIXED FUNCTION ---


try:
    model, X_test, y_test = load_resources()
    # Call the function, passing 'model' as the argument
    prediction_agent, triage_agent, explanation_agent = get_agents(model)
    
except Exception as e:
    st.error(f"⚠️ Error loading resources or agents: {e}")
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

# --- CALLBACK FUNCTION (Ensures state is updated immediately before rerun) ---
def reset_monitoring_state():
    """Reset state to resume monitoring immediately and clean up fraud data."""
    st.session_state.fraud_detected = False
    st.session_state.monitoring = True
    st.session_state.fraud_data = None
    if 'ai_analysis' in st.session_state:
        del st.session_state.ai_analysis
    
    # Advance the index significantly to skip the recently flagged transaction 
    X_test_len = len(X_test)
    st.session_state.current_index = (st.session_state.current_index + 10) % X_test_len


# Header (Static - always visible)
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    # Line 1: Main Title (rendered as H1)
    st.markdown("<h1>🚨 Real-Time Credit Card Fraud Detection Agent</h1>", unsafe_allow_html=True)
    # Line 2: Subtitle (also rendered as H1 for size uniformity)
    st.markdown("<h1>     [Multi-Agent System]</h1>", unsafe_allow_html=True)
with header_col2:
    ticker_placeholder = st.empty()
    if 'ticker_display' not in st.session_state:
        st.session_state.ticker_display = ""

st.markdown("---")

# --- SIDEBAR LOGIC ---
with st.sidebar:
    # Add vertical spacing to move content down
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    st.header("Control Panel")
    
    # 1. State: FRAUD DETECTED (Highest priority, bright Resume button)
    if st.session_state.fraud_detected:
        st.error("FRAUD DETECTED")
        
        fraud_tx = st.session_state.fraud_data['transaction']
        fraud_prob = st.session_state.fraud_data['probability']
        
        # Display fraud details in the sidebar
        st.metric("Card Number", f"**** **** **** {np.random.randint(1000, 9999)}")
        st.metric("Transaction Amount", f"${fraud_tx['Amount']:.2f}")
        st.metric("Fraud Score", f"{fraud_prob:.1%}")
        st.markdown("---")

        # Resume button: Use 'primary' type for a bright, urgent button style
        st.button(
            "▶ RESUME MONITORING", 
            type="primary", 
            use_container_width=True, 
            key="resume_monitoring_sidebar_btn",
            on_click=reset_monitoring_state
        )
        
    # 2. State: MONITORING ACTIVE (Show muted Pause button)
    elif st.session_state.monitoring:
        st.info("Monitoring Active...")
        
        # Pause button: Use 'secondary' type for a muted button style
        if st.button("⏸ PAUSE MONITORING", use_container_width=True, type="secondary"): 
            st.session_state.monitoring = False
            st.rerun()

    # 3. State: INITIAL / PAUSED (Show bright Start/Resume button)
    else: # monitoring is False and fraud_detected is False (Initial or Paused state)
        is_paused = st.session_state.total_processed > 0
        
        button_label = "▶ RESUME DETECTION" if is_paused else "🚨 BEGIN FRAUD DETECTION"
        
        # Start/Resume button: Use 'primary' type for a bright button style
        if st.button(button_label, type="primary", use_container_width=True):
            st.session_state.monitoring = True
            if not is_paused:
                # Reset only on fresh start
                st.session_state.current_index = 0
                st.session_state.total_processed = 0
                st.session_state.fraud_count = 0
            st.rerun()

# --- MAIN CONTENT PLACEHOLDER ---
main_placeholder = st.empty()

with main_placeholder.container():
    
    # 1. START SCREEN (Empty, as button moved to sidebar)
    if not st.session_state.monitoring and not st.session_state.fraud_detected:
        st.markdown("## Ready to begin monitoring?")
        st.markdown("Click the **'🚨 BEGIN FRAUD DETECTION'** button in the sidebar to start processing the transaction stream.")


    # 2. FRAUD REPORT SCREEN (Visible only when fraud_detected is True)
    elif st.session_state.fraud_detected:
        st.error("🚨 FRAUD DETECTED - REVIEW REQUIRED")
        
        # Data pulled from session state
        fraud_tx = st.session_state.fraud_data['transaction']
        fraud_prob = st.session_state.fraud_data['probability']
        
        # --- AGENT CALLS: TRIAGE AGENT determines the action and risk description ---
        action, risk_raw = triage_agent.assess_risk(fraud_prob) 
        
        # Format Risk Level for Display
        if risk_raw == "EXTREMELY HIGH":
            risk_level_display = "🔴 EXTREMELY HIGH - BLOCK IMMEDIATELY"
        elif risk_raw == "HIGH":
            risk_level_display = "🟠 HIGH - ALERT & MANUAL REVIEW"
        else:
            risk_level_display = "🟡 MEDIUM - MONITOR CLOSELY"
        # --- END AGENT CALLS ---

        # Display risk and report in main content
        st.markdown(f"## **Risk Level:** {risk_level_display}")
        st.markdown("---")
        
        st.subheader("🤖 AI Fraud Agent Report")
        
        # Generate Analysis via Explanation Agent
        if 'ai_analysis' not in st.session_state:
            with st.spinner("🔍 Explanation Agent generating report..."):
                if explanation_agent:
                    # Explanation Agent generates the report using the data and Triage result
                    analysis = explanation_agent.generate_analysis(
                        fraud_tx=fraud_tx, 
                        fraud_score=fraud_prob, 
                        action=action, 
                        risk=risk_raw,
                        max_retries=1 
                    )
                    st.session_state.ai_analysis = analysis
                else:
                    st.session_state.ai_analysis = "AI analysis unavailable (ExplanationAgent is disabled due to missing API Key)."
        
        # Display analysis
        st.markdown(st.session_state.ai_analysis)
        st.markdown("---")
        
        # Note: The "RESUME MONITORING" button is now in the sidebar

    # 3. MONITORING SCREEN (Visible only when monitoring is True)
    elif st.session_state.monitoring:
        st.success("✅ MONITORING ACTIVE - Processing transactions...")
        
        # Stats display (FIX 1: Use placeholders to prevent metric layout shifting)
        col1, col2, col3 = st.columns(3)
        
        # Initialize placeholders
        metric_ph1 = col1.empty()
        metric_ph2 = col2.empty()
        metric_ph3 = col3.empty()
        
        # Update metrics (This happens once per rerun)
        metric_ph1.metric("Transactions Processed", st.session_state.total_processed)
        metric_ph2.metric("Frauds Detected", st.session_state.fraud_count)
        metric_ph3.metric("Processing Speed", "50 tx/sec") 
        
        
        # Processing Loop (Simulated Real-Time)
        # FIX 2: Process 50 transactions per rerun to reduce jarring updates
        for i in range(50): 
            if st.session_state.current_index >= len(X_test):
                st.session_state.current_index = 0
            
            tx = X_test.iloc[st.session_state.current_index]
            
            # --- AGENT CALL: Prediction Agent scores the transaction ---
            prob, _ = prediction_agent.score_transaction(tx)
            # --- END AGENT CALL ---

            # Update ticker (in the header placeholder)
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
            
            # Use the high risk threshold from the Triage Agent's rules (0.95)
            if prob >= triage_agent.rules['EXTREMELY HIGH']:
                # FRAUD DETECTED!
                st.session_state.fraud_detected = True
                st.session_state.monitoring = False
                st.session_state.fraud_count += 1
                st.session_state.fraud_data = {
                    'transaction': tx.to_dict(),
                    'probability': prob
                }
                
                # Audio alert
                st.markdown(
                    f"""
                    <audio id="fraud-alert" autoplay>
                        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTGH0fPTgjMGHm7A7+OZSA0PVa3n7qlXEwpBmuL0wm0gBjKM0vPVgzQGH2q/7uGaRANUqzk7qVUEgw+lODzvWsgBzWS1PPZgjIGH2q/7uCZThENUark7qRUEQw9lODztmwiBjiM0vPXgzIFIG7A7uCZTQ4NUark7qRUEgtAmeD0t2wiBjiR1fPYgjQIG6/7+GaTQ8MUqvm7qVUEwxBl+D0uG4iB" type="audio/wav">
                    </audio>
                    <script>
                        // Use a short timeout to ensure the element is rendered before attempting to play
                        setTimeout(() => {{
                            const audio = document.getElementById('fraud-alert');
                            if (audio && typeof audio.play === 'function') {{
                                audio.play().catch(e => console.log("Autoplay blocked:", e));
                            }}
                        }}, 100);
                    </script>
                    """, unsafe_allow_html=True)
                
                time.sleep(0.2)
                st.rerun()
            
            time.sleep(0.01) # Reduced delay for faster simulation (1 tx per 10ms)
        
        # Continue monitoring loop until fraud is detected
        st.rerun()

# Footer
st.markdown("---")
st.caption("🎓 Google Agents Intensive Capstone Project: AI-Powered Credit Card Fraud Detection | XGBoost Model (AUC: 0.9886) | **Multi-Agent Architecture**")
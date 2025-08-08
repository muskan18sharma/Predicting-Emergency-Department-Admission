import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the trained model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = joblib.load('random_forest_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
            
        return model, preprocessor, feature_info
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None

# Main app
def main():
    st.set_page_config(
        page_title="ED Daily Admissions Predictor", 
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Emergency Department Daily Admissions Predictor")
    st.markdown("### Predict daily ED admissions using Random Forest model")
    
    # Introduction section
    st.markdown("---")
    st.subheader("👋 Welcome to the ED Admissions Prediction System")
    
    intro_col1, intro_col2 = st.columns([2, 1])
    
    with intro_col1:
        st.markdown("""
        **Hello! I'm excited to present this Emergency Department Daily Admissions Predictor.**
        
        This application leverages **machine learning** to forecast daily admission volumes in emergency departments, 
        helping healthcare administrators make **data-driven operational decisions**.
        
        **What this app does:**
        🎯 **Predicts** the number of patients likely to be admitted to the hospital from the ED on any given day
        
        📊 **Analyzes** 19 different factors including historical patterns, weather conditions, and operational metrics
        
        🚨 **Categorizes** predictions into Low, Moderate, or High volume levels for easy operational planning
        
        📈 **Utilizes** a Random Forest regression model trained on real healthcare data to provide accurate forecasts
        
        **How to use:** Simply adjust the input features in the sidebar and click the predict button to get your forecast!
        """)
    
    with intro_col2:
        st.info("""
        **Key Benefits:**
        
        ⚡ **Real-time predictions**
        
        📋 **Operational planning**
        
        👥 **Staffing optimization**
        
        🏥 **Resource allocation**
        
        📊 **Data-driven decisions**
        """)
    
    st.markdown("---")
    
    # Load model and preprocessor
    model, preprocessor, feature_info = load_model_and_preprocessor()
    if model is None:
        st.stop()
    
    # Sidebar for all input features
    st.sidebar.header("📊 All Input Features")
    st.sidebar.markdown("---")
    
    # Get feature lists
    numeric_feats = feature_info['numeric_feats']
    categorical_feats = feature_info['categorical_feats']
    
    # Collect inputs
    inputs = {}
    
    # Numeric features in sidebar
    st.sidebar.subheader("🔢 Numeric Features")
    
    # Reorder to put is_patient_admit_within_target after admit_target_lag_3
    reordered_numeric = []
    for feature in numeric_feats:
        if feature != 'is_patient_admit_within_target':
            reordered_numeric.append(feature)
            # Insert is_patient_admit_within_target after admit_target_lag_3
            if feature == 'admit_target_lag_3':
                reordered_numeric.append('is_patient_admit_within_target')
    
    for i, feature in enumerate(reordered_numeric, 1):
        if feature == 'is_patient_admit_within_target':
            # Special handling - rename
            inputs[feature] = st.sidebar.slider(
                f"Input Feature {i}: Admit Target Lag 4",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.01
            )
        elif feature == 'month':
            inputs[feature] = st.sidebar.selectbox(
                f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                options=list(range(1, 13)),
                index=7
            )
        elif feature == 'day_of_week_num':
            inputs[feature] = st.sidebar.selectbox(
                f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                options=list(range(7)),
                index=3,
                format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x]
            )
        elif feature in ['is_weekend', 'is_stat_holiday_bc']:
            inputs[feature] = st.sidebar.selectbox(
                f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                options=[0, 1],
                index=0,
                format_func=lambda x: 'Yes' if x == 1 else 'No'
            )
        elif feature == 'bc_population':
            inputs[feature] = st.sidebar.number_input(
                f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                min_value=5000000,
                max_value=6000000,
                value=5700000,
                step=10000
            )
        elif 'lag' in feature or 'roll_mean' in feature:
            if 'admit_target' in feature:
                # Target rate features (0-1)
                inputs[feature] = st.sidebar.slider(
                    f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.01
                )
            else:
                # Admission count features
                inputs[feature] = st.sidebar.number_input(
                    f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=200.0,
                    value=50.0,
                    step=1.0
                )
        else:
            # Default numeric input
            inputs[feature] = st.sidebar.number_input(
                f"Input Feature {i}: {feature.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0
            )
    
    # Categorical features
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌤️ Categorical Features")
    
    feature_count = len(reordered_numeric) + 1
    for feature in categorical_feats:
        if feature == 'weather_type':
            inputs[feature] = st.sidebar.selectbox(
                f"Input Feature {feature_count}: Weather Type",
                options=['Clear', 'Rainy', 'Snowy'],
                index=0
            )
    
    # Main content area
    if st.button("🔍 **Predict Daily Admissions**", type="primary", use_container_width=True):
        try:
            # Create input DataFrame with exact feature order from training
            all_features = numeric_feats + categorical_feats
            input_data = {feature: inputs[feature] for feature in all_features}
            input_df = pd.DataFrame([input_data], columns=all_features)
            
            # Transform using the saved preprocessor
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_transformed)[0]
            
            # Display results
            st.success("🎯 **Prediction Successful!**")
            
            # Results display
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    "**Predicted Daily Admissions**",
                    f"**{prediction:.0f}**"
                )
            
            with result_col2:
                # Categorize prediction level
                if prediction <= 40:
                    status = "🟢 **Low Volume**"
                elif prediction <= 70:
                    status = "🟡 **Moderate Volume**"
                else:
                    status = "🔴 **High Volume**"
                
                st.metric("**Volume Category**", status)
            
            with result_col3:
                # Calculate capacity utilization
                max_capacity = 100
                utilization = (prediction / max_capacity) * 100
                st.metric(
                    "**Estimated Utilization**",
                    f"**{utilization:.1f}%**"
                )
            
            # Add explanation of volume categories
            st.markdown("---")
            st.subheader("📊 Volume Category Explanation")
            
            explain_col1, explain_col2, explain_col3 = st.columns(3)
            
            with explain_col1:
                st.success("""
                **🟢 Low Volume (≤40 admissions)**
                
                • Manageable patient load
                • Standard staffing adequate
                • Shorter wait times
                • Good patient care capacity
                • Opportunity for training
                """)
            
            with explain_col2:
                st.warning("""
                **🟡 Moderate Volume (41-70 admissions)**
                
                • Increased patient load
                • May need additional staff
                • Moderate wait times
                • Standard protocols apply
                • Monitor capacity closely
                """)
            
            with explain_col3:
                st.error("""
                **🔴 High Volume (>70 admissions)**
                
                • Heavy patient load
                • Extra staffing required
                • Longer wait times expected
                • Activate surge protocols
                • Consider diversions
                """)
            
            # Add explanation of lagged variables and what they represent
            st.markdown("---")
            st.subheader("📊 Understanding Lagged Variables")
            st.info("""
            **Lagged variables capture historical patterns to improve predictions:**
            
            ⏮️ **Lag Features (1, 2, 3, 7 days)**:
            • **Lag 1**: Yesterday's admissions - immediate recent pattern
            • **Lag 2**: 2 days ago - short-term trend continuation  
            • **Lag 3**: 3 days ago - establishes recent directional trend
            • **Lag 7**: 1 week ago - captures weekly cyclical patterns
            
            📊 **Rolling Mean Features**:
            • **Roll Mean 3**: 3-day average - smooths out daily noise
            • **Roll Mean 7**: 7-day average - identifies longer-term trends
            
            🎯 **Admit Target Lag Features**:
            • **Admit Target Lag 1-7**: Historical rates of meeting admission time targets
            • **Admit Target Lag 4**: Current admission target performance rate
            • **Roll Mean 3/7**: Average target performance over time
            
            **Why Lag Variables Matter**:
            📈 Hospital admissions show **temporal dependencies** - today's volume is influenced by recent days
            🔄 **Weekly patterns** - certain days typically busier (captured by lag 7)
            📊 **Trend detection** - rising/falling admission patterns affect future predictions
            🎯 **Performance continuity** - recent target rate performance indicates system efficiency
            """)
            
        except Exception as e:
            st.error(f"❌ **Error making prediction**: {str(e)}")

if __name__ == "__main__":
    main()
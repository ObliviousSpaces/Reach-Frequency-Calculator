import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pygam
from pygam import LinearGAM, s
import math
from scipy.sparse import issparse

# Retrieve the version of pygam
pygam_version = pygam.__version__

# Display the version in your Streamlit app
st.write(f"pygam version: {pygam_version}")

# Cache the data load function so it isn’t re-run unnecessarily
@st.cache_data
def load_data():
    df = pd.read_excel(r"CombinedDataV3.xlsx", sheet_name='CombinedData')
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])
    cols = ['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    # Log transformations
    df['Log_Impressions'] = np.log1p(df['Impressions'])
    df['Log_Audience'] = np.log1p(df['Audience Size'])
    df['Log_Flight'] = np.log1p(df['Flight Period'])
    df['Log_Reach'] = np.log1p(df['Reach'])
    df['Log_Frequency'] = np.log1p(df['Frequency'])
    df['Log_Frequency Cap Per Flight'] = np.log1p(df['Frequency Cap Per Flight'])
    return df

# Cache model training since it's time consuming
@st.cache_resource
def train_models(df):
    # Use full dataset for training (no train/test split)
    X = df[['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']]
    y_reach = df['Log_Reach']
    y_frequency = df['Log_Frequency']
    
    # Convert X to a dense NumPy array if it's sparse, else to a NumPy array
    if issparse(X):
        X = X.toarray()
    else:
        X = X.to_numpy()
    
    # Train Random Forest models
    reach_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    reach_model_rf.fit(X, y_reach)
    
    freq_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    freq_model_rf.fit(X, y_frequency)
    
    # Train GAM models
    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3), verbose=False)
    gam_reach.fit(X, y_reach)
    
    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3), verbose=False)
    gam_freq.fit(X, y_frequency)
    
    return reach_model_rf, freq_model_rf, gam_reach, gam_freq

def calculate_frequency_cap(frequency_input, option, flight_period):
    if option == "Day":
        return frequency_input * flight_period
    elif option == "Week":
        return math.ceil(flight_period / 7) * frequency_input
    elif option == "Month":
        return (flight_period / 30) * frequency_input
    elif option == "Life":
        return frequency_input
    else:
        raise ValueError("Invalid option for frequency cap input.")

def predict_metrics(impressions, audience_size, flight_period, frequency_cap,
                    reach_model_rf, freq_model_rf, gam_reach, gam_freq):
    log_impressions = np.log1p(impressions)
    log_audience = np.log1p(audience_size)
    log_flight = np.log1p(flight_period)
    log_frequency_cap = np.log1p(frequency_cap)
    
    # Prepare input data using the log-transformed values
    input_data = pd.DataFrame(
        [[log_impressions, log_audience, log_flight, log_frequency_cap]],
        columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']
    )
    
    # Make predictions using both models
    log_predicted_reach_rf = reach_model_rf.predict(input_data)[0]
    log_predicted_freq_rf = freq_model_rf.predict(input_data)[0]
    log_predicted_reach_gam = gam_reach.predict(input_data)[0]
    log_predicted_freq_gam = gam_freq.predict(input_data)[0]
    
    # Reverse the log transformation
    predicted_reach_rf = np.expm1(log_predicted_reach_rf)
    predicted_frequency_rf = np.expm1(log_predicted_freq_rf)
    predicted_reach_gam = np.expm1(log_predicted_reach_gam)
    predicted_frequency_gam = np.expm1(log_predicted_freq_gam)
    
    return predicted_reach_rf, predicted_frequency_rf, predicted_reach_gam, predicted_frequency_gam

def main():
    st.title("Media Metrics Prediction App")
    st.write("This app trains models on the full dataset (no test split) and predicts Reach and Frequency metrics.")
    
    # Load data and train models on the full dataset
    df = load_data()
    reach_model_rf, freq_model_rf, gam_reach, gam_freq = train_models(df)
    
    st.subheader("Enter Your Data:")
    impressions = st.number_input("Impression Volume", min_value=0.0, value=1000.0, step=100.0)
    audience_size = st.number_input("Audience Size", min_value=0.0, value=500.0, step=50.0)
    flight_period = st.number_input("Flight Period (days)", min_value=1.0, value=30.0, step=1.0)
    frequency_option = st.selectbox("Frequency Cap Option", ["Day", "Week", "Month", "Life"])
    frequency_input = st.number_input(f"Frequency Cap per {frequency_option}", min_value=0.0, value=5.0, step=1.0)
    
    if st.button("Predict Metrics"):
        frequency_cap = calculate_frequency_cap(frequency_input, frequency_option, flight_period)
        pred_reach_rf, pred_freq_rf, pred_reach_gam, pred_freq_gam = predict_metrics(
            impressions, audience_size, flight_period, frequency_cap,
            reach_model_rf, freq_model_rf, gam_reach, gam_freq
        )
        calculated_frequency_rf = impressions / pred_reach_rf if pred_reach_rf != 0 else 0
        calculated_frequency_gam = impressions / pred_reach_gam if pred_reach_gam != 0 else 0
        
        st.write("### Random Forest Model Predictions")
        st.write(f"Predicted Reach: {pred_reach_rf:.2f}")
        st.write(f"Predicted Frequency: {pred_freq_rf:.2f}")
        st.write(f"Calculated Frequency (Impressions / Predicted Reach): {calculated_frequency_rf:.2f}")
        
        st.write("### GAM Model Predictions")
        st.write(f"Predicted Reach: {pred_reach_gam:.2f}")
        st.write(f"Predicted Frequency: {pred_freq_gam:.2f}")
        st.write(f"Calculated Frequency (Impressions / Predicted Reach): {calculated_frequency_gam:.2f}")

if __name__ == "__main__":
    main()

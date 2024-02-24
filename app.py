import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np

# Load pre-trained models
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('gb_model.pkl', 'rb') as file:
    gb_model = pickle.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('voting_model.pkl', 'rb') as file:
    voting_model = pickle.load(file)

# Streamlit app
def main():
    st.title("Predictive Maintenance App")

    # Sidebar with user input
    st.sidebar.header("User Input")

    # Example: select a file for prediction
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file for prediction", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df_input = pd.read_excel(uploaded_file)  # Use read_excel for Excel files
    
        features = st.sidebar.multiselect("Select features for prediction", ['Temperature', 'Humidity', 'Pressure', 'Vibration'])
        Temperature = st.slider("Temperature", min_value=5.00, max_value=45.00, value=15.0)
        Humidity = st.slider("Humidity", min_value=10.00, max_value=100.00, value=15.0)
        Pressure = st.slider("Pressure", min_value=60.00, max_value=15.00, value=15.0)
        Vibration = st.slider("Vibration", min_value=0.00, max_value=1.00, value=15.0)

        
        if st.sidebar.button("Generate Predictions"):
            # Prepare the input data
            X_input = df_input[features]

            # Make predictions
            rf_pred = rf_model.predict(X_input)
            gb_pred = gb_model.predict(X_input)
            svm_pred = svm_model.predict(X_input)
            voting_pred = voting_model.predict(X_input)

            # Display predictions
            st.subheader("Predictions:")
            st.write("Random Forest Prediction:", rf_pred)
            st.write("Gradient Boosting Prediction:", gb_pred)
            st.write("SVM Prediction:", svm_pred)
            st.write("Voting Regressor Prediction:", voting_pred)

    else:
        st.warning("Please upload an Excel file for prediction.")

if __name__ == "__main__":
    main()
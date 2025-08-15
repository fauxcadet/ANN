import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained ANN model
model = tf.keras.models.load_model('ann_model.h5')

# Load the pickle files for preprocessing
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)    
    with open('label_encoder_geo.pkl', 'rb') as file: # Corrected variable name from geo to gender
        label_encoder_gender = pickle.load(file)
    with open('ohe_geo.pkl', 'rb') as file:
        one_hot_encoder_geo = pickle.load(file)    
except FileNotFoundError as e:
    st.error(f"Error loading a file: {e}. Please ensure 'ann_model.h5', 'scaler.pkl', 'label_encoder_gender.pkl', and 'ohe_geo.pkl' are in the same directory.")
    st.stop()


# Streamlit app layout
st.title("Bank Customer Churn Prediction")    
st.write("Enter customer details to predict if they will churn.")

# User input widgets
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 40) # Added a default value for better UX
balance = st.number_input('Balance', min_value=0.0, max_value=1000000.0, step=1000.0, value=50000.0)
credit_score = st.number_input('Credit Score', value=650)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=1000000.0, step=1000.0, value=60000.0) 
tenure = st.slider('Tenure', 0, 10, 5)
number_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create a button to trigger prediction
if st.button('Predict Churn'):

    # Prepare input data as a dictionary
    input_data_dict = {
       'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [number_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
    
    # Corrected logic:
    # 1. Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame(input_data_dict)

    # 2. One-hot encode the 'Geography' input separately
    geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
    
    # 3. Concatenate the two DataFrames. This is where your original code had an error.
    # We now concatenate two DataFrames instead of a dictionary and a DataFrame.
    final_input_df = pd.concat([input_df, geo_encoded_df], axis=1)

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(final_input_df)

    # Make the prediction
    prediction = model.predict(input_scaled)
    prediction_prob = prediction[0][0]

    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result")
    if prediction_prob > 0.5:
        st.error(f"The customer is likely to leave the bank. (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"The customer is likely to stay with the bank. (Probability: {1 - prediction_prob:.2f})")


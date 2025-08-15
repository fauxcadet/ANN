import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

model=tf.keras.models.load_model('ann_model.h5')

## loading the pickjle files
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)    
with open('label_encoder_geo.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('ohe_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)    


## stream lit app 
st.title("Bank Customer Churn Prediction")    

## user inpit
geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0, max_value=1000000.0, step=1000.0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=1000000.0, step=1000.0) 
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = ({
   'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
##    'Geography': [geography]
})

## one hot encode ''GEography'input
geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)
## scale the input data
input_scaled = scaler.transform(input_data)
## prediction churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]
if prediction_prob > 0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")

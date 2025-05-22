import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

# Load all the pickle files
model = load_model('model.h5')

with open('LE_Gender.pkl', 'rb') as file:
    LE_Gender = pickle.load(file)

with open('OHE_Geography.pkl', 'rb') as file:
    OHE_Geo = pickle.load(file)

with open('Scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)


st.title("Customer Churn Prediction")

geography = st.selectbox('Geogaphy', OHE_Geo.categories_[0])
gender = st.selectbox('Gender', LE_Gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])


# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

print(input_data)

input_data['Gender'] = LE_Gender.transform(input_data['Gender'])


geo_encoded = OHE_Geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = OHE_Geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis = 1)

input_data_scaled = scalar.transform(input_data)

prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

st.write(f'Prediction probability: {pred_prob: .2f}')

if pred_prob > 0.5:
    st.write("The customer is likely to Churn.")
else:
    st.write("The customer is not likely to Churn.")
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
model = tf.keras.models.load_model('final_model_ann.keras')
with open('label_enc_gen.pkl','rb') as file:
    label_enc_gen = pickle.load(file)
with open('ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)
with open('sc.pkl','rb') as file:
    sc = pickle.load(file)
#streamlit
st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_enc_gen.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit_Score')
estimated_salary = st.number_input('Estimated_Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_enc_gen.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member ],
    "EstimatedSalary":[estimated_salary]

})
geo_encode = ohe_geo.transform([[geography]]).toarray()
geo_encode_df = pd.DataFrame(geo_encode,columns=ohe_geo.get_feature_names_out(['Geography']))
# combine one
input_data = pd.concat([input_data.reset_index(drop=True),geo_encode_df], axis=1)
#scaleing data
input_data_scaled = sc.transform(input_data)
#predict Churn
prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

if pred_prob > 0.5 : 
    st.write('The Customer is likely to Stay')
else : 
    st.write('The Customer is unlikely to stay')

#Streamlit UI
import streamlit as st
import pandas as pd
#Model de-serialization
import pickle
with open("Linear_model.pkl", "rb") as f:
    model = pickle.load(f)

#model.predict(data)

#Encoder de-serialization
with open("label_encoder.pkl", "rb") as f1:
    encoder = pickle.load(f1)

#load the cleaned data
df = pd.read_csv('cleaned_data.csv')

st.set_page_config(page_title="Bangalore House Price Prediction", page_icon=":house:", layout="centered")
with st.sidebar:
    st.title("Bangalore House Price Prediction")
    

#imput fields
location =st.selectbox("Location", options=df['location'].unique())
bhk = st.selectbox("BHK", options=df['bhk'].unique())
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)
bath = st.selectbox("Number of Restrooms", options=df['bath'].unique())

#encoding the location
encoded_loc = encoder.transform([location])

#new data preparation
new_data = [[bhk,sqft,bath,encoded_loc[0]]]
#prediction
col1, col2 = st.columns([1,2])
if col2.button("Predict Price"):
    prediction = model.predict(new_data)[0]
    prediction = round(prediction*100000, 2)
    st.subheader(f"Predicted Price: ₹ {prediction}")


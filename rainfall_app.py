# Streamlit Web application for Rainfall Prediction - Kajal Mahore

# Import necessary packages
import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Create a header
st.set_page_config(page_title ='Weather Condition - Kajal')

# Add a title to application in body
st.title("Weather Condition Prediction - Kajal Mahore")

# Take rainfall as input from user
rainfall_input = st.number_input("rainfall (mm):", min_value=0.0, max_value=500.0)

# Take temperature as input from user
temperature_input = st.number_input("temperature (degree celsius):", min_value=-20.0, max_value=60.0)

# Take humidity as input from user
humidity_input = st.number_input("humidity (%):", min_value=0.0, max_value=100.0)

# Take wind_speed as input from user
wind_speed_input = st.number_input("wind_speed (km/h):", min_value=0.0, max_value=200.0)

# Create a predict button
submit = st.button('Predict')

st.subheader('Predictions are : ')

# Create a function to predict the weather_condition and probability of prediction
def predict_Weather(pipe_path, model_path):
    # Construct a dataframe from inputs
    dct = {'rainfall': [rainfall_input],
            'temperature': [temperature_input],
            'humidity': [humidity_input],
            'wind_speed': [wind_speed_input]}
    xnew = pd.DataFrame(dct)
    # Load the pipeline from the Rainfall notebook folder
    with open(pipe_path, 'rb') as file1:
        pre = pickle.load(file1)
     # load the model
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    # Preprocess the xnew
    xnew_pre = pre.transform(xnew)
    # Get predictions 
    pred = model.predict(xnew_pre)
    # Get probability
    prob = model.predict_proba(xnew_pre)
    # Get max probability
    max_prob = np.max(prob)
    return pred, max_prob

# Logic if i press submit button
if submit:
    model_path = "Rainfall notebook/model.pkl"
    pipe_path = "Rainfall notebook/pipe.pkl"
    pred, max_prob = predict_Weather(pipe_path, model_path)
    st.subheader(f'Predicted Weather is: {pred[0]}')
    st.subheader(f'Probability of prediction: {max_prob:.4f}')
    st.progress(max_prob)






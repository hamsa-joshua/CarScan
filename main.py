import numpy as np
import pandas as pd
import pickle as pk 
import streamlit as st 

# Load the model
model = pk.load(open('cpm.pkl', 'rb'))

st.header('Used Car Price Prediction')

# Load the data
cars_data = pd.read_csv('cars.csv')

def brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip() 

cars_data['name'] = cars_data['name'].apply(brand_name)

# Input widgets
name = st.selectbox('Select Brand', cars_data['name'].unique())
year = st.slider('Car Manufacturing Year', 1994, 2024)
km_driven = st.slider('Kms Driven', 15, 200000)
fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission', cars_data['transmission'].unique())
owner = st.selectbox('Owner Type', cars_data['owner'].unique())  # Fixed duplicate label
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Car Engine', 700, 5000)
max_power = st.slider('Car Power', 50, 300)
seats = st.slider('Car Seats', 4, 10)

if st.button('Predict Price'):
    # Create input data as a dictionary first
    input_dict = {
        'name': name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }
    
    # Convert to DataFrame
    input_data_model = pd.DataFrame([input_dict])
    
    # Apply the same transformations as in your model training
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], 
                               [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Petrol', 'Diesel', 'CNG', 'LPG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'], 
       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], inplace=True)
    
    # Make prediction
    car_price = model.predict(input_data_model)
    st.markdown(f'Price of the car is {car_price[0]:.2f}')  # Formatting the price
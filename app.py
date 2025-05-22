from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('cpm.pkl', 'rb'))
cars_data = pd.read_csv('cars.csv')

def brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(brand_name)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to populate select options
@app.route('/get-options')
def get_options():
    options = {
        'name': sorted(cars_data['name'].unique().tolist()),
        'fuel': sorted(cars_data['fuel'].unique().tolist()),
        'seller_type': sorted(cars_data['seller_type'].unique().tolist()),
        'transmission': sorted(cars_data['transmission'].unique().tolist()),
        'owner': sorted(cars_data['owner'].unique().tolist()),
    }
    return jsonify(options)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_dict = {
        'name': data['name'],
        'year': int(data['year']),
        'km_driven': int(data['km_driven']),
        'fuel': data['fuel'],
        'seller_type': data['seller_type'],
        'transmission': data['transmission'],
        'owner': data['owner'],
        'mileage': float(data['mileage']),
        'engine': float(data['engine']),
        'max_power': float(data['max_power']),
        'seats': int(data['seats'])
    }

    df = pd.DataFrame([input_dict])

    df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], 
                        [1, 2, 3, 4, 5], inplace=True)
    df['fuel'].replace(['Petrol', 'Diesel', 'CNG', 'LPG'], [1, 2, 3, 4], inplace=True)
    df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    df['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                        'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                        'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                        'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                        'Ambassador', 'Ashok', 'Isuzu', 'Opel'], 
                        list(range(1, 32)), inplace=True)

    price = model.predict(df)[0]
    return jsonify({'price': price})

if __name__ == '__main__':
    app.run(debug=True)

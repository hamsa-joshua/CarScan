import streamlit as st
import pandas as pd
import pickle as pk
import json
from PIL import Image
from ultralytics import YOLO
import tempfile

# Load ML models
price_model = pk.load(open('cpm.pkl', 'rb'))
yolo_model = YOLO("runs/detect/train/weights/best.pt")

# Load data
cars_data = pd.read_csv('cars.csv')
with open('car_parts_value_map.json') as f:
    part_cost_map = json.load(f)

# Clean brand names
def brand_name(car_name):
    return car_name.split(' ')[0].strip()
cars_data['name'] = cars_data['name'].apply(brand_name)

# Streamlit UI
st.title("🚗 Used Car Price Prediction + Damage Detection")
st.header("1️⃣ Car Details")

name = st.selectbox('Select Brand', cars_data['name'].unique())
year = st.slider('Car Manufacturing Year', 1994, 2024)
km_driven = st.slider('Kms Driven', 15, 200000)
fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission', cars_data['transmission'].unique())
owner = st.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Car Engine (CC)', 700, 5000)
max_power = st.slider('Car Max Power (bhp)', 50, 300)
seats = st.slider('Number of Seats', 4, 10)

# Predict price
car_price = 0
if st.button('💰 Predict Price'):
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
    df = pd.DataFrame([input_dict])

    # Transform
    df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    df['fuel'].replace(['Petrol', 'Diesel', 'CNG', 'LPG'], [1, 2, 3, 4], inplace=True)
    df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    df['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
           'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
           'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
           'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
           'Ambassador', 'Ashok', 'Isuzu', 'Opel'], 
           list(range(1, 32)), inplace=True)

    car_price = price_model.predict(df)[0]
    st.success(f"Predicted Price (No Damage): ₹{car_price:.2f}")

st.header("2️⃣ Upload Image for Damage Detection")
uploaded_file = st.file_uploader("Upload an image of the car", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        result = yolo_model.predict(temp.name, save=False, imgsz=416, conf=0.25)

        damaged_parts = []
        for r in result:
            if r.boxes:
                classes = r.boxes.cls.tolist()
                for cls_id in classes:
                    part_name = yolo_model.names[int(cls_id)]
                    damaged_parts.append(part_name)

        if damaged_parts:
            res_plotted = result[0].plot()
            st.image(res_plotted, caption="Damage Detection Result", use_column_width=True)

            st.subheader("🧾 Detected Damaged Parts:")
            damage_cost = 0
            for part in damaged_parts:
                cost = part_cost_map.get(name, {}).get(part, 0)
                st.write(f"- {part}: ₹{cost}")
                damage_cost += cost

            st.warning(f"Total Damage Cost: ₹{damage_cost:.2f}")

            if car_price > 0:
                final_price = car_price - damage_cost
                st.success(f"💰 Final Adjusted Price: ₹{final_price:.2f}")
            else:
                st.info("Please predict the price first to see adjusted value.")
        else:
            st.info("✅ No visible damage detected.")

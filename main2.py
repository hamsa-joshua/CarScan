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

# Initialize session state for damage info
if 'damage_cost' not in st.session_state:
    st.session_state.damage_cost = 0
    st.session_state.damaged_parts = []
    st.session_state.detection_done = False

# Section 1: Car Details
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

# Section 2: Damage Detection
st.header("2️⃣ Upload Image for Damage Detection")
uploaded_file = st.file_uploader("Upload an image of the car", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        result = yolo_model.predict(temp.name, save=False, imgsz=416, conf=0.25)

        st.session_state.damaged_parts = []
        for r in result:
            if r.boxes:
                classes = r.boxes.cls.tolist()
                for cls_id in classes:
                    part_name = yolo_model.names[int(cls_id)]
                    st.session_state.damaged_parts.append(part_name)

        if st.session_state.damaged_parts:
            res_plotted = result[0].plot()
            st.image(res_plotted, caption="Damage Detection Result", use_column_width=True)

            st.subheader("🧾 Detected Damaged Parts:")
            st.session_state.damage_cost = 0
            for part in st.session_state.damaged_parts:
                cost = part_cost_map.get(name, {}).get(part, 0)
                st.write(f"- {part}: ₹{cost}")
                st.session_state.damage_cost += cost

            st.warning(f"Total Damage Cost: ₹{st.session_state.damage_cost:.2f}")
            st.session_state.detection_done = True
        else:
            st.info("✅ No visible damage detected.")
            st.session_state.damage_cost = 0
            st.session_state.detection_done = True

# Section 3: Final Price Prediction
st.header("3️⃣ Final Price Prediction")

if st.button('💰 Predict Final Price'):
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

    car_price = price_model.predict(df)[0]
    st.success(f"Base Price (No Damage): ₹{car_price:.2f}")
    
    if st.session_state.detection_done:
        final_price = car_price - st.session_state.damage_cost
        st.success(f"💰 Final Adjusted Price: ₹{final_price:.2f}")
        
        if st.session_state.damaged_parts:
            st.subheader("Damage Breakdown:")
            for part in st.session_state.damaged_parts:
                cost = part_cost_map.get(name, {}).get(part, 0)
                st.write(f"- {part}: -₹{cost}")
            st.write(f"Total Damage Deduction: -₹{st.session_state.damage_cost:.2f}")
    else:
        st.info("ℹ️ No damage detection performed. Using base price.")
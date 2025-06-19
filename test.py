import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# Load the YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # Update path if needed

st.title("🚗 Car Damage Detection")

uploaded_file = st.file_uploader("Upload an image of the car", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        result = model.predict(temp.name, save=False, imgsz=416, conf=0.25)

        # Display the result
        for r in result:
            if r.boxes is not None and len(r.boxes) > 0:
                res_plotted = r.plot()  # draw boxes
                st.image(res_plotted, caption="Detection Result", use_column_width=True)
            else:
                st.warning("No damage detected.")


import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import joblib
import os
import requests

# === Load model and scaler ===
from model import DualResNetWithMetadata  # Make sure model.py is in the same folder

MODEL_PATH = "best_model.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1W3q6MDSmSBUcAiFmC9QfxQaYzAPj-Pky"
SCALER_PATH = "scaler.pkl"
LABEL_MAP = {0: "fun_vibe_here", 1: "i_love_it_here", 2: "not_my_vibe", 3: "why_god_why"}

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualResNetWithMetadata(tabular_dim=5, num_classes=4)  # Adjust dim if needed
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Remove unwanted keys if loading full checkpoint
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

scaler = joblib.load(SCALER_PATH)

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Streamlit UI ===
st.title("🌍 Vibe Classifier")
st.write("Upload two images (Sentinel + OSM) and optionally add metadata to classify the vibe.")

# === Image Uploads ===
sentinel_file = st.file_uploader("Upload Sentinel Image (Satellite View)", type=["jpg", "jpeg", "png"])
osm_file = st.file_uploader("Upload OSM Image (Street View or Google Place)", type=["jpg", "jpeg", "png"])

st.subheader("Optional Metadata")
traffic_level = st.selectbox("Traffic Level", ["low", "medium", "high"], index=1)
time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"], index=1)
road_density = st.slider("Road Density (0–300)", 0, 300, 120)
dist_to_park = st.number_input("Distance to Nearest Park (in meters)", 0, 2000, 300)

# === Inference Logic ===
if sentinel_file and osm_file:
    sentinel_image = Image.open(sentinel_file).convert("RGB")
    osm_image = Image.open(osm_file).convert("RGB")

    st.image(sentinel_image, caption="🛰️ Sentinel Image", use_column_width=True)
    st.image(osm_image, caption="🗺️ OSM / Place Image", use_column_width=True)

    if st.button("🔮 Predict Vibe"):
        # === Image tensors ===
        sentinel_tensor = transform(sentinel_image).unsqueeze(0).to(device)
        osm_tensor = transform(osm_image).unsqueeze(0).to(device)

        # === Tabular metadata features ===
        road_length = road_density * 50 + np.random.normal(10, 5)  # Proxy computation
        populartimes_peak_avg = np.random.randint(30, 100)  # Simulated value


        # Match training features exactly
        traffic_onehot = [
            int(traffic_level == "low"),
            int(traffic_level == "medium"),
        ]  # traffic_level_high is implicitly False

        tabular_raw = np.array([[ 
            dist_to_park,
            road_density,
            road_length,
            *traffic_onehot
        ]])


        tabular_scaled = scaler.transform(tabular_raw)
        tabular_tensor = torch.tensor(tabular_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(sentinel_tensor, osm_tensor, tabular_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        st.success(f"🎯 Predicted Vibe: **{LABEL_MAP[pred_idx]}**")
        st.write(f"Confidence: `{confidence:.2%}`")

else:
    st.warning("Please upload both Sentinel and OSM images to proceed.")

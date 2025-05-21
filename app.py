import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import joblib
import os
import gdown
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# === Load model and scaler ===
from model import DualResNetWithMetadata  # Make sure model.py is in the same folder

MODEL_PATH = "best_model.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1W3q6MDSmSBUcAiFmC9QfxQaYzAPj-Pky"
SCALER_PATH = "scaler.pkl"
LABEL_MAP = {0: "fun_vibe_here", 1: "i_love_it_here", 2: "not_my_vibe", 3: "why_god_why"}

if not os.path.exists(MODEL_PATH):
    print("Downloading model via gdown...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
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

# === Vibe Classes Map from CSV ===
st.markdown("---")
st.header("🗺️ Vibe Map")

DATA_PATH = "datasets/vibe_full_features.csv"

try:
    df = pd.read_csv(DATA_PATH)
    # Adjust column names here to your csv exactly
    df = df.rename(columns=lambda x: x.strip())  # remove whitespace if any
    # Using exact names you gave: lat, lon, vibe_class (adjust if different)
    lat_col = "lat"
    lon_col = "lon"
    vibe_col = "vibe_class"
    if all(col in df.columns for col in [lat_col, lon_col, vibe_col]):
        df = df[[lat_col, lon_col, vibe_col]].dropna()

        # Simple color map for vibe classes - add more if needed
        color_map = {
            "fun_vibe_here": "green",
            "i_love_it_here": "blue",
            "not_my_vibe": "orange",
            "why_god_why": "red"
        }

        center = [df[lat_col].mean(), df[lon_col].mean()]
        m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df.iterrows():
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f"Vibe: {row[vibe_col]}",
                icon=folium.Icon(color=color_map.get(row[vibe_col], "gray"))
            ).add_to(marker_cluster)

        st_folium(m, width=1000, height=700)

    else:
        st.error(f"CSV must contain columns: '{lat_col}', '{lon_col}', '{vibe_col}'")
except FileNotFoundError:
    st.error(f"File not found: `{DATA_PATH}`")
except Exception as e:
    st.error(f"Error loading CSV data: {e}")

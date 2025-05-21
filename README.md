# 🛰️ VIBE_PLACE_MAPPER – Multimodal Vibe Classification App

This project classifies the **vibe of a location** (e.g., "fun vibe here", "why god why") using a unique **multimodal deep learning pipeline** that fuses:
- 🗺️ **Satellite imagery (Sentinel-2 RGB)**
- 🏙️ **Street-level imagery (Google Place / OSM)**
- 📊 **Contextual metadata** like road density, traffic level, and more.

Built with PyTorch, scikit-learn, GEE (Google Earth Engine), and deployed via Streamlit Cloud.

---

## 📂 Folder Structure

```
📁 VIBE_PLACE_MAPPER/
├── app.py                     # Streamlit web app
├── model.py                   # Dual-ResNet + MLP model definition
├── classifier.ipynb           # ResNet-based classifier training + evaluation
├── data_prep.ipynb            # Data download: Google, OSM, TomTom, IMD
├── feature_engineering.ipynb  # Boruta-based feature selection
├── sentinel_download_gee.ipynb # Download Sentinel-2 RGB tiles via GEE
├── best_model.pt              # Trained model checkpoint
├── scaler.pkl                 # StandardScaler fitted on tabular features
├── requirements.txt           # Project dependencies
├── .gitignore                 # Excludes heavy data/image files from Git
├── datasets/                  # Contains vibe_full_features.csv and images
├── checkpoints/, cache/, results/  # Optional logging/output folders
```

---

## 📊 Data Sources

- **Sentinel-2 RGB Imagery** from Google Earth Engine (`COPERNICUS/S2_SR`)
- **Street-level Images** from Google Places API and OSM
- **Traffic + Popular Times** from TomTom & Google APIs
- **Metadata** like road length, park proximity via OSMnx
- **Weather** from IMD (manual integration)

---

## 🤖 Model Architecture

- Dual **ResNet-50** branches: one for Sentinel image, one for OSM image
- A separate **MLP** branch processes contextual metadata
- Features are fused and passed to a **final classifier head**
- Trained using **cross-entropy loss**, evaluated with accuracy, F1-score, etc.

---

## 🧪 Notebooks

- `data_prep.ipynb`: Collects data using Google, TomTom, IMD, OSM APIs
- `feature_engineering.ipynb`: Applies Boruta to select best features
- `sentinel_download_gee.ipynb`: Exports Sentinel-2 tiles using GEE
- `classifier.ipynb`: Trains and evaluates the full multimodal classifier

---

## 🚀 Web Deployment

Streamlit frontend (`app.py`) allows users to:
- Upload Sentinel + OSM image pairs
- Provide optional metadata (traffic, road density, etc.)
- Predict the vibe class with confidence score

🖥 **Try the app**: [https://vibe-classifier.streamlit.app](https://vibe-classifier.streamlit.app)  
📦 **GitHub repo**: [https://github.com/Dharani1207/Vibe-Classifier.git](https://github.com/Dharani1207/Vibe-Classifier.git)

---

## 📌 Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

For Earth Engine access, authenticate:
```bash
earthengine authenticate
```

---





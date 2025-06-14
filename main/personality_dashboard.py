import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Model Definition ---
class PersonalityModel(nn.Module):
    def __init__(self, input_size=7, hidden1=128, hidden2=64, output_size=2):  # <- match here
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x):
        return self.net(x)
    
    
# --- Load Model ---
model = PersonalityModel()
model.load_state_dict(torch.load("E:\computer vision\personality_model.pt"))
model.eval()

# --- Label Encoder Setup ---
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Ambivert", "Extrovert", "Introvert"])  # Update as needed

# --- Streamlit UI ---
st.set_page_config(page_title="Personality Prediction Dashboard", layout="centered")
st.title("🧠 Personality Prediction Web App")
st.write("Enter the personality traits to get a prediction.")

# Input sliders
openness = st.slider("Openness", 0.0, 10.0, 5.0)
conscientiousness = st.slider("Conscientiousness", 0.0, 10.0, 5.0)
extroversion = st.slider("Extroversion", 0.0, 10.0, 5.0)
agreeableness = st.slider("Agreeableness", 0.0, 10.0, 5.0)
neuroticism = st.slider("Neuroticism", 0.0, 10.0, 5.0)
social_support = st.slider("Social Support", 0.0, 10.0, 5.0)
self_esteem = st.slider("Self-Esteem", 0.0, 10.0, 5.0)

# Predict
if st.button("Predict Personality"):
    features = torch.tensor([[openness, conscientiousness, extroversion, agreeableness,
                              neuroticism, social_support, self_esteem]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(features)
        pred_idx = torch.argmax(outputs, dim=1).item()
        personality = label_encoder.inverse_transform([pred_idx])[0]
    st.success(f"### 🎯 Predicted Personality: {personality}")

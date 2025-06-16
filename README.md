# PersonalityPulse: Predicting Introvert vs. Extrovert Using Deep Learning

This project uses a deep learning model in PyTorch to predict whether a person is an introvert or an extrovert based on their personality traits. The model is trained on the [Personality Prediction Dataset](https://www.kaggle.com/datasets/shalmamuji/personality-prediction-data-introvert-extrovert) from Kaggle.

## 🚀 Features
- Trained on real psychometric data
- Deep Neural Network built with PyTorch
- Streamlit dashboard for real-time predictions
- High test accuracy (~99%)
- Feature importance analysis using permutation method

## 📂 Dataset
This Kaggle dataset includes personality factors like:
- Openness
- Conscientiousness
- Extroversion
- Agreeableness
- Neuroticism
- Self-esteem
- Social support

## 🎥 Output Video

[*(Replace with actual link to video output or upload `.mp4` to GitHub repo if small enough)*](https://github.com/user-attachments/assets/8b243889-e030-4859-b9c7-a0bdb42ec6d6)

📎 Source: [Kaggle Dataset](https://www.kaggle.com/datasets/shalmamuji/personality-prediction-data-introvert-extrovert)

## 🧠 Model Architecture
```python
Input (7 features)
→ Linear (128) → ReLU → Dropout(0.3)
→ Linear (64) → ReLU → Dropout(0.3)
→ Linear (2 outputs: Introvert / Extrovert)

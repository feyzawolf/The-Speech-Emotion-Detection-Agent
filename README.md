# 🎙️ Speech Emotion Recognition using Azure

## 📌 Overview
This project focuses on Speech Emotion Recognition (SER), where the goal is to detect human emotions from audio signals.

The system uses machine learning (SVM + MFCC features) and is deployed on Azure, with integration into an Azure Function App and Azure AI Foundry for intelligent responses.

---

## 🎯 Objective
The goal of this project is to:
- Detect emotions from speech audio (e.g. happy, sad, angry, neutral)
- Compare performance between dataset audio and real-life recordings
- Build a scalable cloud-based solution using Azure services
- Integrate emotion detection into an intelligent agent via Azure AI Foundry

---

## 🏗️ System Architecture

### Dataset-based flow:
1. Audio file uploaded to Azure Blob Storage
2. Event Grid triggers the Azure Function
3. Function extracts features and predicts emotion
4. Result is stored and retrieved via API

### Live audio flow:
- Audio is recorded from the browser
- Sent directly to the Azure Function
- Prediction happens instantly (Blob Storage is skipped)

---

## 🤖 Model Details
- Model: Support Vector Machine (SVM)
- Feature extraction: MFCC (Mel-Frequency Cepstral Coefficients)
- Dataset: CREMA-D + additional custom recordings
- Output: Emotion label (e.g. happy, sad, angry, neutral)

---

## ☁️ Azure Services Used
- Azure Functions → backend API for predictions
- Azure Blob Storage → stores uploaded audio files
- Azure Event Grid → triggers processing automatically
- Azure AI Foundry → connects predictions to an intelligent agent

---

## 📊 Results & Insights
- High accuracy on dataset audio
- Lower accuracy on real-world recordings
- Model tends to bias towards "angry" for unseen voices

👉 This highlights a key challenge:
Models trained on controlled datasets do not generalize well to real-world scenarios.

---

## ⚠️ Limitations
- Limited diversity in training data (voices, gender, tone)
- Audio quality differences (dataset vs browser recordings)
- No dedicated validation set (evaluation done manually)
- Bias in predictions (e.g. over-predicting "angry")

---

## 🔧 Improvements (Future Work)
- Add more diverse speakers (different genders, accents)
- Use a proper validation set
- Try deep learning models (e.g. CNN, LSTM)
- Improve preprocessing for live audio
- Fine-tune integration with Azure AI Foundry

---

## 💡 Key Learning
This project shows that:
- Real-world performance is more complex than dataset accuracy
- Cloud integration (Azure) enables scalable deployment
- Combining ML with AI platforms (Foundry) creates more intelligent systems

---

## ⚙️ Installation

Install dependencies with:

pip install -r requirements.txt

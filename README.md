# 🧠 Muscle Dystrophy Detection Using Voice

## 📌 Overview
Muscle dystrophy is a group of genetic disorders that cause muscle weakness and degeneration. This project uses **deep learning (CNN + LSTM)** to analyze voice recordings and classify muscle dystrophy severity levels.

## 📂 Dataset
The dataset consists of speech recordings labeled with severity levels. It includes:
- **Training Set**: 500 audio samples
- **Test Set**: 200 audio samples
- **Validation Set**: 150 audio samples

Audio features are extracted using **MFCC (Mel-Frequency Cepstral Coefficients)** for model training.

## 🔍 Model Architecture
- **CNN** for feature extraction from MFCCs
- **LSTM** for analyzing temporal dependencies in speech patterns
- **Softmax Classifier** for severity level classification

### 🏗️ Model Pipeline:
1️⃣ **Preprocessing**: Convert audio to MFCC features  
2️⃣ **Feature Extraction**: Apply CNN for feature learning  
3️⃣ **Sequence Learning**: LSTM processes time-series speech data  
4️⃣ **Classification**: Final layer predicts severity level  

## ⚙️ Installation & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/muscle-dystrophy-detection.git
cd muscle-dystrophy-detection

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Use
### 🏋️‍♂️ Train the Model
```bash
python train.py
```

### 🔍 Predict Severity from an Audio File
```bash
python predict.py --file sample.wav
```

## 📊 Results & Performance
| Model      | Accuracy | Loss  |
|------------|---------|-------|
| CNN+LSTM   | 91.2%   | 0.08  |
| ResNet50   | 87.6%   | 0.12  |

## 🎯 To-Do / Future Enhancements
✅ Improve model accuracy with fine-tuning  
✅ Deploy as a **web app using Streamlit**  
✅ Implement **real-time speech processing**  

## 🤝 Contributing
We welcome contributions! If you’d like to improve this project:
1. Fork the repo
2. Create a new branch (`feature-improvement`)
3. Commit and push your changes
4. Submit a pull request

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
For any questions or collaboration, reach out to [bhuvanjitendar@gmail.com].

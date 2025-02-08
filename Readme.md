# Speech Emotion Recognition using LSTM

## 📌 Project Overview
This project implements a **Speech Emotion Recognition (SER) System** using the **Toronto Emotional Speech Set (TESS)** dataset. It extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features from audio files and uses an **LSTM-based neural network** to classify emotions.

## 📂 Dataset
The dataset used is **Toronto Emotional Speech Set (TESS)**, which consists of speech samples categorized into different emotions.

🔗 **Dataset Source:** [Kaggle - TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## 📌 Project Workflow
1. **Dataset Loading**: Download and extract the dataset.
2. **Exploratory Data Analysis (EDA)**: Visualize waveform and spectrogram of different emotions.
3. **Feature Extraction**: Extract MFCC features from speech samples.
4. **Data Preprocessing**: One-hot encode the labels and reshape features for LSTM input.
5. **Model Building**: Train an LSTM-based neural network for classification.
6. **Training & Evaluation**: Train the model and visualize accuracy/loss curves.

## 🛠️ Tech Stack
- Python
- Pandas & NumPy
- Librosa
- Seaborn & Matplotlib
- Scikit-Learn
- Keras & TensorFlow

## 📜 Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### **2️⃣ Install Dependencies**
```sh
pip install numpy pandas librosa seaborn matplotlib tensorflow keras
```

### **3️⃣ Run the Script**
```sh
python main.py
```

## 🔍 Data Visualization
### **Waveform Plot**
Visualizing the waveform of speech samples for different emotions.
```python
waveplot(data, sampling_rate, emotion)
```

### **Spectrogram Plot**
Visualizing spectrograms to analyze frequency components.
```python
spectogram(data, sampling_rate, emotion)
```

## 🎯 Model Architecture
- **LSTM Layer**: Extracts sequential features from MFCCs.
- **Dropout Layers**: Prevents overfitting.
- **Dense Layers**: Fully connected layers for classification.
- **Softmax Activation**: Outputs probability distribution across emotions.

```python
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
```

## 📊 Model Performance
The model is trained for **50 epochs** with **batch size 64**, and training progress is visualized using accuracy and loss curves.

## 📌 Future Improvements
✅ Implement **data augmentation** (time shifting, noise addition, pitch shifting).  
✅ Optimize **hyperparameters** using Grid Search or Bayesian Optimization.  
✅ Deploy as a **web app** using Flask or Streamlit.  

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License
This project is licensed under the **MIT License**.

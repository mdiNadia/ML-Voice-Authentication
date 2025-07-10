
# 🎙️ Voice Authentication & Speaker Classification System

This project is a **machine learning-based voice authentication system** developed as the final course project for **MSc Artificial Intelligence** at **University of Tehran (2025)**. It focuses on **speaker identification**, **gender classification**, and **voice clustering** using advanced **audio signal processing** and **deep learning** techniques.

---

## 📌 Project Goals

- ✅ Build an **intelligent system** to classify speaker identity and gender using voice samples.
- ✅ Compare **open-set** and **closed-set authentication** mechanisms.
- ✅ Extract powerful audio features (MFCC, Log-Mel, Spectral Contrast, etc.).
- ✅ Handle real-world challenges like **noise**, **voice spoofing**, and **emotion-based voice variation**.
- ✅ Apply **ML/DL models** to identify and cluster voices effectively.

---

## 🧠 Features & Techniques

### 🎧 Audio Preprocessing
- Band-pass filtering (100–8000 Hz)
- LUFS normalization
- Silence removal using RMS
- Duration trimming

### 🔍 Feature Extraction
- **Log-Mel Spectrogram**
- **MFCC (Mel Frequency Cepstral Coefficients)**
- Spectral Centroid & Contrast
- Zero Crossing Rate (ZCR)
- LPC / PLP
- Energy features (frame-wise)

### 🧪 Machine Learning Models
- Support Vector Machines (SVM)
- KMeans & GMM (for clustering)
- PCA / Feature Reduction
- Feature Selection & Validation (Silhouette Score, Cross-validation)

### 🤖 Deep Learning
- CNNs for spectrogram image classification
- RNNs (LSTM/GRU) for time-series voice patterns
- Contrastive Loss & Embedding Adaptation for open-set

---

## 🛠 Tech Stack

| Category | Tools / Libraries |
|---------|-------------------|
| Language | Python |
| Audio | `librosa`, `soundfile`, `pyloudnorm` |
| ML / DL | `scikit-learn`, `PyTorch`, `Torch`, `Pandas`, `NumPy` |
| Visualization | `Matplotlib`, `Seaborn` |
| Evaluation | `Silhouette Score`, `Confusion Matrix`, `Entropy Threshold` |

---

## 📁 Project Structure

```
ml_project/
├── Audio_Scripts/
│   ├── preprocessing.py         # Noise reduction, silence removal
│   ├── feature_extraction.py    # MFCC, Log-Mel, spectral features
│   └── audio_utils.py           # Dataset utilities
├── Audio_GenderDetection/       # Gender classification
├── Audio_Authentication/        # Speaker recognition (open/closed set)
├── Audio_Clustering/            # Clustering voice features
└── data/
    └── raw/                     # ~473 labeled voice files
```

---

## 📊 Dataset Summary

- **473 audio samples** (~35.5 hours)
- **114 speakers** (86 male, 28 female)
- Files manually curated, normalized, and labeled
- Each speaker: 3× 3-second voice clips for training/testing
- Clustering evaluated using **Silhouette Score**

---

## 🎯 Results & Highlights

- Achieved high accuracy in gender classification via MFCC & Log-Mel.
- Successfully implemented **open-set authentication** using similarity learning.
- Applied **deep learning** models (CNN, GRU) with meaningful performance gains.
- Preprocessed audio signals show improved model robustness in noisy conditions.

---

## 🧪 Sample Visualizations

<p align="center">
  <img src="examples/mfcc_example.png" width="400"/> 
  <img src="examples/logmel_example.png" width="400"/>
</p>

---

## 👥 Contributors

- **Omid Molaei**
- **Nadie Mohammadi**
- **Fatemeh Sadri**

Supervised by: *MSc AI – Faculty of Engineering, University of Tehran*

---

## 📌 How to Use

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/voice-auth-ml.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run gender detection:
   ```bash
   python Audio_GenderDetection/gender_model.py
   ```
4. For clustering:
   ```bash
   python Audio_Clustering/kmeans_clustering.py
   ```

---

## 📜 License

MIT License — for academic and research use.

# 🎤 Emotion Recognition from Voice using MFCC and SVM

This project uses machine learning to recognize human emotions from voice recordings by extracting MFCC features and training a Support Vector Machine (SVM) classifier.

---

## 🧠 Objective

To build a speech emotion recognition system using audio features from the **RAVDESS** dataset. The goal is to classify emotions such as *happy*, *sad*, *angry*, *calm*, *fearful*, etc., based on vocal patterns using MFCC (Mel-Frequency Cepstral Coefficients).

---

## 📁 Dataset

* **Source**: [RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
* Contains `.wav` files labeled with emotion codes in their filenames.
* Emotions supported:

  * Neutral
  * Calm
  * Happy
  * Sad
  * Angry
  * Fearful
  * Disgust
  * Surprised

---

## ⚙️ How it works

1. **Filename Parsing**: Emotions are decoded from the third component of the filename.
2. **Feature Extraction**: MFCCs are extracted from audio files using `librosa`.
3. **Data Preprocessing**:

   * Invalid files and unknown emotions are skipped.
   * Features are averaged across time for consistency.
4. **Label Encoding**: Emotion labels are encoded to numeric values using `LabelEncoder`.
5. **Model Training**: A linear SVM classifier is trained on the extracted features.
6. **Evaluation**: The model is evaluated using accuracy and a classification report.

---

## 🧪 Output

* **Model**: SVM (Support Vector Classifier) with a linear kernel
* **Evaluation Metrics**:

  * Accuracy
  * Precision, Recall, F1-score for each emotion
* Example Output:

  ```
  Accuracy: 0.87
  Classification Report:
                precision    recall  f1-score   support
       angry       0.86       0.89       0.87       25
        happy       0.90       0.84       0.87       25
        sad         0.88       0.88       0.88       25
        ...
  ```

---

## 🧰 Tech Stack

* **Python**
* **Libraries**:

  * `Librosa` – Audio feature extraction
  * `Scikit-learn` – ML model training and evaluation
  * `NumPy`, `Glob`, `os` – Data handling

---

## 📂 File Structure

```
├── audio_project/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── ...
├── emotion_recognition.py  ← Your main script
```

---

## 🚀 How to Run

1. Place all RAVDESS `.wav` files inside `audio_project/Actor_XX` folders.
2. Update the path in `base_audio_dir` to match your local directory.
3. Run the script:

   ```bash
   python emotion_recognition.py
   ```
4. Check console output for accuracy and classification report.

---

## 👩‍💻 Author

**Kavya Kapoor**
📧 [kavyakapoor869@gmail.com](mailto:kavyakapoor869@gmail.com)


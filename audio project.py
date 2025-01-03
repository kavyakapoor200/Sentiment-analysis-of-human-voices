
import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Define the path to the base dataset directory (parent folder containing actor folders)
base_audio_dir = r'C:\Users\pc\Documents\audio project'

# Collect all audio files recursively from all actor folders
audio_files = glob.glob(os.path.join(base_audio_dir, '**', '*.wav'), recursive=True)

# Print total number of audio files
print(f"Total audio files found: {len(audio_files)}")

# Emotion mapping based on RAVDESS filename format
emotion_mapping = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract emotion label based on the filename
def get_emotion_from_filename(file_name):
    parts = file_name.split('-')  # Split the filename into parts based on '-'
    if len(parts) >= 3:  # Ensure the filename has enough parts
        emotion_code = parts[2]  # Emotion code is the 3rd part
        emotion = emotion_mapping.get(emotion_code, 'unknown')  # Map emotion code to label
        return emotion
    else:
        print(f"Filename {file_name} doesn't match expected format.")
        return None  # Return None if the format is incorrect

# Function to extract MFCC features from an audio file
def extract_mfcc_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load the audio file
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features (13 coefficients)
        return np.mean(mfcc, axis=1)  # Return the mean of the MFCCs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None  # Return None if there is an issue processing the file

# Extract features and labels from all files
features = []
labels = []

for file_path in audio_files:
    file_name = os.path.basename(file_path)
    emotion = get_emotion_from_filename(file_name)  # Extract emotion

    if emotion is None or emotion == 'unknown':
        print(f"Skipping file with invalid format: {file_name}")
        continue  # Skip files with invalid format

    mfcc_features = extract_mfcc_features(file_path)  # Extract features
    if mfcc_features is not None:
        features.append(mfcc_features)
        labels.append(emotion)
    else:
        print(f"Skipping file due to feature extraction error: {file_name}")

# Check the number of features and labels
print(f"Total features extracted: {len(features)}")
print(f"Total labels extracted: {len(labels)}")

# If no features are extracted, exit gracefully
if len(features) == 0:
    print("No valid audio features extracted. Exiting.")
else:
    # Convert labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

    # Train a classifier (Support Vector Classifier in this case)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

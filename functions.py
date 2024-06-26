import pandas as pd
import numpy as np
# import keras
import pickle
import librosa

from keras.models import load_model


# model_file = open('model.pkl', 'rb')
encoder_file = open('encoder.pkl', 'rb')

# model = pickle.load(open('model.pkl', 'rb'))
model = load_model('model.h5')
encoder = pickle.load(encoder_file)

# Function to extract MFCC features from audio
def extract_features(data, sample_rate):
  # 1. MFCC
  mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)

  # 2. chroma_stft
  stft =  np.abs(librosa.stft(data))
  chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

  # 3. Mel spectogram
  mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

  # 4. zero crossing rate
  zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, frame_length=2048, hop_length=512).T, axis=0)

  # 5. RMSE
  rmse = np.mean(librosa.feature.rms(y=data, frame_length=2048, hop_length=512).T, axis=0)

  features = np.hstack([mfcc, chroma_stft, mel, zcr, rmse])

  return features

# Function to load audio, extract features, and apply augmentations
def get_features(path):
  data, sample_rate = librosa.load(path)

  # Extract features from the base audio
  base = extract_features(data, sample_rate)

  return pd.DataFrame(base)

def generate_df(path):
    features_list = []
    features = get_features(path)
    features_list.append(features)

    all_features = pd.concat(features_list, ignore_index=True)

    final_df = all_features.T
    return final_df

def predict_result(data):
    pred_test = model.predict(data)
    Y_pred = encoder.inverse_transform(pred_test)
    return Y_pred
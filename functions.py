import pandas as pd
import numpy as np
import pickle
import librosa
import streamlit as st
import matplotlib.pyplot as plt
# import keras

from keras.models import load_model
from sklearn.model_selection import train_test_split

model = load_model('pkl/model.h5')
encoder = pickle.load(open('pkl/encoder.pkl', 'rb'))
# history = pickle.load(open('history.pkl', 'rb'))
with open('pkl/history.pkl', 'rb') as file:
    history = pickle.load(file)

# Function to save uploaded audio file
def save_uploaded_file(uploaded_file):
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_audio_file"

# Function to process audio file
def process_audio(file_path):
    try:
        df = generate_df(file_path)
        # st.write(df)
        result = predict_result(df)
        return result
    except Exception as e:
        st.error(f"Error processing the audio file: {e}")

def user_evaluation(predicted_emotion, target_emotion):
    if predicted_emotion == target_emotion:
        st.write("Congratulations! Your acting was perfect!")
    else:
        st.write("Don't worry, with more practice, you'll master it! Keep going!")

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
    return Y_pred[0][0]

def model_history():
    st.write("Accuracy of our model on test data :",  86.21593117713928)
    # Extract training history
    train_acc = history['accuracy']
    train_loss = history['loss']
    test_acc = history['val_accuracy']
    test_loss = history['val_loss']

    epochs = np.arange(len(train_acc))

    # Create the plots
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    # Plot training and testing loss
    ax[0].plot(epochs, train_loss, label='Training Loss')
    ax[0].plot(epochs, test_loss, label='Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    # Plot training and testing accuracy
    ax[1].plot(epochs, train_acc, label='Training Accuracy')
    ax[1].plot(epochs, test_acc, label='Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

    # Display the plots in Streamlit
    st.pyplot(fig)
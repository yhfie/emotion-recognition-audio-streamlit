import streamlit as st
import pandas as pd
import numpy as np
# import tensorflow as tf
# import keras
from st_audiorec import st_audiorec
# import librosa
import tempfile

from functions import generate_df, predict_result, save_uploaded_file, process_audio

from st_audiorec import st_audiorec

# st.write(tf.__version__)
# st.write(librosa.__version__)
# st.write(keras.__version__)

acting_script = pd.read_csv('acting-scripts.csv')

# Initialize session state variables
shuffle = st.button("Shuffle")
st.write("Act on following script and emotions!")
if ('rand_script' not in st.session_state) or shuffle:
    acting_script = pd.read_csv('acting-scripts.csv')
    rand_row = acting_script.sample()
    st.session_state.rand_script = rand_row["script"].values[0]
    st.session_state.rand_emotion = rand_row["emotion"].values[0]

# Use session state variables
st.title(st.session_state.rand_script)
st.write("Emotion: ", st.session_state.rand_emotion)
st.divider()
st.subheader("Record audio")
audio_record = st_audiorec()
st.write("or")
st.subheader("Upload audio")
audio_upload = st.file_uploader("Upload file", type=['wav', 'mp3', 'flac'])

st.divider()

# Process uploaded file
if audio_upload:
    st.write("Your audio:")
    st.audio(audio_upload, format='audio/wav')
    predict = st.button("Proceed")
    if predict:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_upload.read())
            temp_file_path = temp_file.name
        process_audio(temp_file_path)

# Process recorded audio
if audio_record:
    st.write("Your audio:")
    st.audio(audio_record)
    predict = st.button("Proceed")
    if predict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_record)
            temp_file_path = temp_file.name
        process_audio(temp_file_path)



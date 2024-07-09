import streamlit as st
import pandas as pd
import numpy as np
# import tensorflow as tf
# import keras
from st_audiorec import st_audiorec
import librosa

from functions import generate_df, predict_result

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
audio_upload = st.file_uploader(" ")

st.divider()

# if audio_upload is not None:
#     st.audio(audio_upload, format='audio/wav')

# if audio_record is not None:
#     st.audio(audio_record, format='audio/wav')

if audio_upload:
    st.write("Your audio:")
    st.audio(audio_upload, format='audio/wav')
    predict = st.button("Proceed")
    if predict:
        df = generate_df(audio_upload)
        st.write(df)
        result = predict_result(df)
        st.write("result: ", result)

if audio_record:
    st.write("Your audio:")
    st.audio(audio_record)
    predict = st.button("Proceed")
    if predict:
            try:
                # Save the recorded audio temporarily
                file_path = "temp_audio_record.wav"
                with open(file_path, "wb") as f:
                    f.write(audio_record)
                # Generate DataFrame and predict result
                df = generate_df(file_path)
                st.write(df)
                result = predict_result(df)
                st.write("Result: ", result)
            except Exception as e:
                st.error(f"Error processing the audio file: {e}")



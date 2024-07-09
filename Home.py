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
if ('rand_script' not in st.session_state) or shuffle:
    acting_script = pd.read_csv('acting-scripts.csv')
    rand_row = acting_script.sample()
    st.session_state.rand_script = rand_row["script"].values[0]
    st.session_state.rand_emotion = rand_row["emotion"].values[0]

# Use session state variables
st.header(st.session_state.rand_script)
st.write("Emotion: ", st.session_state.rand_emotion)

st.write("Record audio")
audio_record = st_audiorec()
st.write("or")
audio_upload = st.file_uploader("Upload file")

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
        df = generate_df(audio_record)
        st.write(df)
        result = predict_result(df)
        st.write("result: ", result)



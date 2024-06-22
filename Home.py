import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from functions import generate_df, predict_result

from st_audiorec import st_audiorec

# st.write(tf.__version__)
# st.write(keras.__version__)

acting_script = pd.read_csv('acting-scripts.csv')

# Initialize session state variables
if 'rand_script' not in st.session_state:
    acting_script = pd.read_csv('acting-scripts.csv')
    rand_row = acting_script.sample()
    st.session_state.rand_script = rand_row["script"].values[0]
    st.session_state.rand_emotion = rand_row["emotion"].values[0]

# Use session state variables
st.header(st.session_state.rand_script)
st.write("Emotion: ", st.session_state.rand_emotion)

# audio_record = st_audiorec()
audio_upload = st.file_uploader("Upload file")

if audio_upload is not None:
    st.audio(audio_upload, format='audio/wav')

if audio_upload:
    predict = st.button("Proceed")
    if predict:
        # if audio_record:
        #     df = generate_df(audio_record)
        df = generate_df(audio_upload)
        st.write(df)
        result = predict_result(df)
        st.write("result: ", result)



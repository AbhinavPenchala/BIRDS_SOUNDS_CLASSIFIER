# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:31:08 2024

@author: pench
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import io

# Load the pre-trained model
with open('C:/Users/pench/OneDrive/Desktop/BIRD_SPECIES_PROJECT/trained_model.sav', 'rb') as f:
    model = pickle.load(f)

# Load bird species
train_data = pd.read_csv('C:/Users/pench/OneDrive/Desktop/birdclef-2024/train_metadata.csv')
bird_species = train_data.primary_label.unique()

# Function to process the uploaded audio file and extract MFCC features
def process_audio_file(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded file into memory
        audio_bytes = uploaded_file.read()
        
        # Use io.BytesIO to handle the audio file
        audio_data = io.BytesIO(audio_bytes)
        
        # Load the audio file
        sound, sr = librosa.load(audio_data, sr=32000, duration=15)
        mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_fft=2048)
        mfcc_mean = mfcc.mean(axis=1)
        
        return mfcc_mean, audio_data
    return None, None

# Streamlit UI
st.set_page_config(page_title="Birds Sounds Classifier", page_icon="C:/Users/pench/OneDrive/Desktop/BIRD_SPECIES_PROJECT/bird2")

# Add logo and title
st.image("C:/Users/pench/OneDrive/Desktop/BIRD_SPECIES_PROJECT/bird1", width=300)
st.title("Bird Species Prediction")
st.write("Upload an audio file to predict the bird species.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["ogg", "wav", "mp3"])

if uploaded_file is not None:
    # Process the uploaded audio file
    mfcc_features, audio_data = process_audio_file(uploaded_file)
    
    if mfcc_features is not None:
        # Make prediction
        mfcc_features = mfcc_features.reshape(1, -1)  # Reshape for prediction
        prediction_probs = model.predict_proba(mfcc_features)
        
        # Get the predicted species name
        predicted_species = bird_species[np.argmax(prediction_probs)]
        
        # Display the prediction
        st.write(f"Predicted Bird Species: **{predicted_species}**")
        
        # Play the uploaded audio file
        st.audio(audio_data, format='audio/ogg')
    else:
        st.write("Error processing audio file.")
else:
    st.write("Please upload an audio file to get started.")

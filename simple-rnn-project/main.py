import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# load the imdb dataset word index

word_index = imdb.get_word_index()
word_index

revrese_word_index = {value: key for (key, value) in word_index.items()}
revrese_word_index

# load the pre-trained model with relu activation

model = load_model('simple_rnn.h5')

model.summary()

# creating helper functions

def decode_review(text):
    return ' '.join([revrese_word_index.get(i -3, '?') for i in text])

def preprocess_text(text):
  words = text.lower().split()
  encoded = [word_index.get(word, 2)+3 for word in words]
  padded = sequence.pad_sequences([encoded], maxlen=500)
  return padded


## predicition function

def predict_review(text):
  preprocessed_text = preprocess_text(text)
  prediction = model.predict(preprocessed_text)
  print(prediction)

  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

  return sentiment, prediction[0][0]


## streamlit app

import streamlit as st

st.header('IMDB Movie Reviews Sentimental Analysis')

st.write('Enter a movie review to classify it is positive or negative')

user_input = st.text_input('Enter the movie review ')

if st.submit('Classify'):
    
    sentiment, score = predict_review(user_input)
    
    st.write('Sentiment: ',{sentiment})
    st.write('Predicition Score: ',{score})
    
else:
    st.write('Please enter the review')

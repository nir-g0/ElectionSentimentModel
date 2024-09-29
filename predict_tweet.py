import pickle
import joblib
import numpy as np
import tensorflow as tf

# Load your trained model and tokenizer
model = joblib.load('model.pkl')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Define preprocessing and prediction functions
def preprocess_tweet(tweet, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([tweet])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

def predict_tweet(tweet):
    processed_tweet = preprocess_tweet(tweet, tokenizer, 100)
    prediction = model.predict(processed_tweet)
    predicted_class = (prediction > 0.5).astype(int)
    return predicted_class[0][0]

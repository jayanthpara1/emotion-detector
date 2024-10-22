import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('emotion_model.h5')

# Define the emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def get_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    reshaped_frame = resized_frame.reshape(1, 48, 48, 1) / 255.0

    predictions = model.predict(reshaped_frame)
    emotion_idx = np.argmax(predictions)
    emotion = emotions[emotion_idx]
    confidence = predictions[0][emotion_idx]

    return emotion, confidence

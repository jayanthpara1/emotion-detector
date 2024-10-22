import cv2
import numpy as np
from keras.models import load_model

model = load_model('model/emotion_model.h5')
emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral', 4: 'Cool', 5: 'Surprised'}

def get_emotion(frame):
    # Preprocess the frame for prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    prediction = model.predict(reshaped)
    emotion_index = np.argmax(prediction)
    return emotion_dict[emotion_index], max(prediction[0])

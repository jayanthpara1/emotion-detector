import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    data = []
    labels = []

    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            img = cv2.resize(img, (48, 48))  # Resize to 48x48
            data.append(img)
            labels.append(emotion_idx)

    data = np.array(data).reshape(-1, 48, 48, 1) / 255.0  # Normalize and reshape
    labels = to_categorical(np.array(labels), len(emotions))
    
    return train_test_split(data, labels, test_size=0.2, random_state=42)

# Example usage:
# X_train, X_test, y_train, y_test = load_data('data/fer2013')

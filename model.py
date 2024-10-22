import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }

    for emotion, label in emotion_map.items():
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, 48, 48, 1) / 255.0  # Normalize images
    labels = to_categorical(np.array(labels), num_classes=len(emotion_map))
    
    return images, labels

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # 7 classes for FER-2013

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Load training data
    X_train, y_train = load_data('data/train')  # Adjust the path to your dataset
    X_test, y_test = load_data('data/test')  # Adjust the path to your dataset

    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save('emotion_model.h5')

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('emotion_model.h5')

# Emotion labels and their corresponding colors
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_colors = {
    'angry': (0, 0, 255),        # Red
    'disgust': (0, 128, 0),      # Green
    'fear': (255, 0, 0),         # Blue
    'happy': (0, 255, 255),      # Yellow
    'sad': (255, 255, 255),      # White
    'surprise': (255, 165, 0),   # Orange
    'neutral': (128, 128, 128)   # Gray
}

def generate_frames():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Process the face for emotion detection
            face_region = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_region, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

            # Predict the emotion
            predictions = model.predict(reshaped_face)[0]
            top_indices = np.argsort(predictions)[-2:][::-1]  # Get top 2 emotions
            top_emotions = [(emotion_labels[i], predictions[i]) for i in top_indices]

            # Highlight the highest percentage emotion
            highest_emotion, highest_confidence = top_emotions[0]
            color = emotion_colors[highest_emotion]
            font_scale = 1 + highest_confidence  # Increase font size based on confidence
            
            cv2.putText(frame, f'{highest_emotion}: {highest_confidence:.2f}', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Display the second emotion
            second_emotion, second_confidence = top_emotions[1]
            cv2.putText(frame, f'{second_emotion}: {second_confidence:.2f}', 
                        (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

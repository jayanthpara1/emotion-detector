from model import build_model
from data_preprocessing import load_data
from keras.callbacks import ModelCheckpoint

# Load the data
X_train, X_test, y_train, y_test = load_data('data/fer2013')

# Build and compile the model
model = build_model()

# Set up a checkpoint to save the best model during training
checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=50, 
                    batch_size=64, 
                    callbacks=[checkpoint])

# Save the final model
model.save('emotion_model_final.h5')

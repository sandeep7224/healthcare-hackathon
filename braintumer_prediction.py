import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

# Define paths to your data directory
data_dir = r'C:\Users\Dell\healthcare\multiple disease predictio\tumerdata'  # Use raw string literal

# Define parameters
img_height, img_width = 150, 150
batch_size = 32
validation_split = 0.2
test_split = 0.1

# Load data and create training, validation, and test datasets
def load_data(data_dir, img_height, img_width, batch_size, validation_split, test_split):
    dataset = image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        labels='inferred',
        label_mode='int',
        validation_split=validation_split,
        subset='training',
        seed=123
    )

    # Split dataset
    val_ds = image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        labels='inferred',
        label_mode='int',
        validation_split=validation_split,
        subset='validation',
        seed=123
    )

    test_ds = dataset.take(int(len(dataset) * test_split))
    train_ds = dataset.skip(int(len(dataset) * test_split))

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = load_data(data_dir, img_height, img_width, batch_size, validation_split, test_split)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    batch_size=batch_size
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.2f}")

# Predict on the test dataset
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred = (y_pred > 0.5).astype(int)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the model
model.save('brain_tumor_model.h5')

# Load the model (if needed)
# model = tf.keras.models.load_model('brain_tumor_model.h5')

# Predict on a new image (example)
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Tumor" if prediction[0] > 0.5 else "No Tumor"

# Example usage
result = predict_image('Image5.jpg')
print(f'Prediction: {result}')


joblib.dump(model,"tumer_save.joblib")

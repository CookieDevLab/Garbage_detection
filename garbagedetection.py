import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Step 1: Set the correct path to extracted images ===
image_dir = r"C:\Users\USER\OneDrive\文档\garbagedetection99.py\garbagedetection.py\extracted_dataset\Images\Images"
images = []
labels = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".png")):
        label = 0 if "clean" in filename.lower() else 1
        img_path = os.path.join(image_dir, filename)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

if images.size == 0 or labels.size == 0:
    raise ValueError("No images found. Check the dataset path and file names.")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

# === Step 2: Define and train the CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
model.save("model.keras")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# === Step 3: Predict a random test image ===
idx = random.randint(0, len(X_test) - 1)
test_img = X_test[idx]
prediction = model.predict(test_img.reshape(1, 224, 224, 3))
label = "Dirty" if prediction[0][0] > 0.5 else "Clean"

plt.imshow(test_img)
plt.title(f"Predicted: {label}")
plt.axis("off")
plt.show()

# === Step 4: Accuracy/Loss Graphs ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()

# === Step 5: Confusion Matrix & Classification Report ===
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int32")

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Clean", "Dirty"]))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Clean", "Dirty"], yticklabels=["Clean", "Dirty"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

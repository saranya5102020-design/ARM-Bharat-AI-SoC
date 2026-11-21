import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# -----------------------------
# 1. Configuration
# -----------------------------

DATA_DIR = Path("PlantVillage_Images/raw/color")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5

# -----------------------------
# 2. Load dataset
# -----------------------------

print("Loading PlantVillage dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)

# Save class names
Path("class_names.txt").write_text(
    "\n".join(class_names),
    encoding="utf-8"
)

# -----------------------------
# 3. Use a smaller subset first
# -----------------------------

train_ds = train_ds.take(200)
val_ds = val_ds.take(40)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Dataset ready!")
print("Training batches:", 200)
print("Validation batches:", 40)

# -----------------------------
# 4. Build CNN model
# -----------------------------

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

# -----------------------------
# 5. Compile model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 6. Train CNN
# -----------------------------

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# 7. Save trained model
# -----------------------------

model.save("crop_disease_cnn.keras")

print("\nCNN training completed!")
print("Model saved as crop_disease_cnn.keras")
import tensorflow as tf
from pathlib import Path

# Dataset settings
DATASET_DIR = Path("PlantVillage_Images/raw/color")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
SEED = 42

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class_names = train_ds.class_names

print("Number of classes:", len(class_names))
print("MobileNetV2 dataset loaded successfully!")
# Step 3: Load pretrained MobileNetV2

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

print("MobileNetV2 pretrained model loaded!")
print("Total layers:", len(base_model.layers))
print("Trainable:", base_model.trainable)
# Step 4: Build the classification head

inputs = tf.keras.Input(shape=(160, 160, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

print("MobileNetV2 classification model created!")
model.summary()
# Step 5: Compile the model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")
# Step 6: Train the classification head

EPOCHS = 5

history = model.fit(
    train_ds.take(500),
    validation_data=val_ds.take(100),
    epochs=EPOCHS
)

print("MobileNetV2 training completed!")
# Step 7: Save model and training history

import json

model.save("crop_disease_mobilenet_v3.keras")

with open("class_names_v3.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

with open("training_history_v3.json", "w") as f:
    json.dump(history.history, f)

print("Model, class names and training history saved successfully!")
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

DATASET_DIR = Path("PlantVillage_Images/raw/color")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
SEED = 42

# Load the complete validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = val_ds.class_names

# Load trained MobileNetV2 model
model = tf.keras.models.load_model(
    "crop_disease_mobilenet_v3.keras"
)

y_true = []
y_pred = []

print("Evaluating all validation images...")

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Classification report
print("\nCNN V3 Classification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

# Overall accuracy
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print("\nOverall Validation Accuracy:", accuracy * 100, "%")
print("Total validation images:", len(y_true))
print("Evaluation completed!")
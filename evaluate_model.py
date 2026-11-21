import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = Path("PlantVillage_Images/raw/color")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42

# Load the same validation split
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class_names = val_ds.class_names

# Load CNN V2
model = tf.keras.models.load_model("crop_disease_cnn_v2.keras")

# Use the same 40 validation batches as CNN V2 training


y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print("Overall Accuracy:", accuracy * 100, "%")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(18, 15))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("CNN V2 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()

plt.savefig("confusion_matrix_v2.png", dpi=150)
plt.show()

print("\nEvaluation completed!")
print("Saved: confusion_matrix_v2.png")
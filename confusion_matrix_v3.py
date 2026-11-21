import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = Path("PlantVillage_Images/raw/color")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
SEED = 42

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
model = tf.keras.models.load_model("crop_disease_mobilenet_v3.keras")

y_true = []
y_pred = []

print("Generating predictions...")

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)
np.save("confusion_matrix_v3_data.npy", cm)
np.save("class_names_v3.npy", np.array(class_names))

print("Confusion matrix data saved!")

short_names = [
    name.replace("___", "\n")
        .replace("_", " ")
        .replace("(including sour)", "")
    for name in class_names
]

plt.figure(figsize=(24, 22))

sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    xticklabels=short_names,
    yticklabels=short_names
)

plt.title("ARM Bharat AI-SoC - MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()

plt.savefig("confusion_matrix_v3_clean.png", dpi=200)
plt.show()

print("Clean confusion matrix generated successfully!")
print("Saved as confusion_matrix_v3_clean.png")
print("\nPER-CLASS ACCURACY")

for i, name in enumerate(class_names):
    total = np.sum(cm[i])
    correct = cm[i][i]

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"{name}: {accuracy:.2f}%")
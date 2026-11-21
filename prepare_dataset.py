import tensorflow as tf
from pathlib import Path

# Dataset location
DATASET_DIR = Path("PlantVillage_Images/raw/color")

# Image settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

print("Loading PlantVillage dataset...")

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
    batch_size=BATCH_SIZE
)

# Save class names
class_names = train_ds.class_names

print("\nNumber of classes:", len(class_names))
print("Classes:", class_names)

print("\nTraining batches:", len(train_ds))
print("Validation batches:", len(val_ds))

# Optimize data loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("\nDataset preparation completed!")
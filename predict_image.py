import tensorflow as tf
import numpy as np

from disease_info import display_advisory

# Load trained CNN model
model = tf.keras.models.load_model("crop_disease_cnn_v2.keras")

# Load class names
with open("class_names.txt", "r", encoding="utf-8") as file:
    class_names = file.read().splitlines()

# Get image path
image_path = input("Enter leaf image path: ").strip().strip('"')

# Load and preprocess image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# CNN Prediction
predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

# Display prediction
print("\nPrediction:", predicted_class)
print("Model confidence:", round(float(confidence), 2), "%")

# Display farmer advisory
display_advisory(predicted_class)
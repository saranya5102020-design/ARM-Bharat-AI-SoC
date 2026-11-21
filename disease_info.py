disease_database = {
    "Tomato Target Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "description": "A leaf disease classification detected by the CNN.",
        "precautions": [
            "Inspect affected plants regularly.",
            "Avoid unnecessary wetting of foliage.",
            "Consult local agricultural experts for treatment."
        ],
        "source": "Agricultural guidance source to be verified"
    },

    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "condition": "Leaf Scorch",
        "description": "Strawberry leaf scorch classification.",
        "precautions": [
            "Inspect nearby plants.",
            "Avoid unnecessary wetting of leaves.",
            "Consult an agricultural expert."
        ],
        "source": "Agricultural guidance needs verification"
    },

    "Unknown": {
        "crop": "Unknown",
        "condition": "Uncertain prediction",
        "description": "The model could not confidently identify the condition.",
        "precautions": [
            "Capture a clear image in good lighting.",
            "Check the plant with an agricultural expert."
        ],
        "source": "General precaution"
    }
}


def get_disease_info(prediction):
    return disease_database.get(prediction, disease_database["Unknown"])


# Automatically extract crop name and condition
def extract_crop_and_condition(prediction):
    parts = prediction.split("___")

    if len(parts) >= 2:
        crop = parts[0].replace("_", " ").strip()
        condition = parts[1].replace("_", " ").strip()

        return crop, condition

    return "Unknown", prediction.replace("_", " ")


# Display farmer advisory
def display_advisory(prediction):

    result = get_disease_info(prediction)

    # If no detailed database entry exists,
    # extract crop and condition from CNN class name
    if result == disease_database["Unknown"]:
        crop, condition = extract_crop_and_condition(prediction)

        result = {
            "crop": crop,
            "condition": condition,
            "description": f"{condition} classification detected by the CNN.",
            "precautions": [
                "Inspect affected plants regularly.",
                "Avoid unnecessary wetting of foliage.",
                "Consult a local agricultural expert."
            ],
            "source": "General agricultural precautions; verify treatment"
        }

    print("\n--- Farmer Advisory ---")
    print("Crop:", result["crop"])
    print("Condition:", result["condition"])
    print("Description:", result["description"])
    print("Precautions:")

    for advice in result["precautions"]:
        print("-", advice)

    print("Source:", result["source"])


# Test the module independently
if __name__ == "__main__":
    test_prediction = "Strawberry___Leaf_scorch"
    display_advisory(test_prediction)
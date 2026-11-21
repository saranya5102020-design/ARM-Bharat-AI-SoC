import ollama

# ==========================================
# 1. DISEASE INFORMATION DATABASE
# ==========================================

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


# ==========================================
# 2. GET DISEASE INFORMATION
# ==========================================

def get_disease_info(prediction):
    return disease_database.get(
        prediction,
        disease_database["Unknown"]
    )


# ==========================================
# 3. EXTRACT CROP AND CONDITION
# ==========================================

def extract_crop_and_condition(prediction):
    parts = prediction.split("___")

    if len(parts) >= 2:
        crop = parts[0].replace("_", " ").strip()
        condition = parts[1].replace("_", " ").strip()

        return crop, condition

    return "Unknown", prediction.replace("_", " ")


# ==========================================
# 4. GENERATE FARMER ADVISORY USING LLAMA
# ==========================================

def generate_llama_advisory(result):

    precautions = "\n".join(
        "- " + item for item in result["precautions"]
    )

    prompt = f"""
Explain the following agricultural information
in simple English for farmers.

Crop: {result["crop"]}
Condition: {result["condition"]}
Description: {result["description"]}

Verified information supplied:
{precautions}

Source: {result["source"]}

IMPORTANT RULES:
1. Use only the supplied information.
2. Do not invent symptoms or disease details.
3. Do not suggest medicines, pesticides or treatments.
4. Do not add unsupported facts.
5. Explain the precautions clearly.
6. If information is uncertain, say so.
"""

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a farmer advisory assistant. "
                        "Your task is only to simplify supplied "
                        "agricultural information. Never invent facts."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        return response["message"]["content"]

    except Exception as error:
        print("\nLlama is currently unavailable.")
        print("Showing database advisory instead.")
        print("Technical details:", error)

        return None


# ==========================================
# 5. DISPLAY COMPLETE FARMER ADVISORY
# ==========================================

def display_advisory(prediction):

    result = get_disease_info(prediction)

    # Handle predictions without a detailed database entry
    if prediction not in disease_database:
        crop, condition = extract_crop_and_condition(prediction)

        result = {
            "crop": crop,
            "condition": condition,
            "description": (
                f"{condition} classification detected by the CNN."
            ),
            "precautions": [
                "Inspect affected plants regularly.",
                "Avoid unnecessary wetting of foliage.",
                "Consult a local agricultural expert."
            ],
            "source": "General agricultural precautions; verify treatment"
        }

    print("\n================================")
    print("       ARM BHARAT AI-SoC")
    print("        FARMER ADVISORY")
    print("================================")

    print("\nCrop:", result["crop"])
    print("Condition:", result["condition"])
    print("Description:", result["description"])

    print("\nDatabase Precautions:")

    for advice in result["precautions"]:
        print("-", advice)

    print("\nInformation Source:", result["source"])

    # Generate explanation using Llama
    print("\nGenerating farmer-friendly explanation...")

    llama_result = generate_llama_advisory(result)

    if llama_result:
        print("\n===== LLAMA FARMER EXPLANATION =====")
        print(llama_result)

    print("\n================================")
    print("End of Farmer Advisory")
    print("================================")


# ==========================================
# 6. INDEPENDENT TEST
# ==========================================

if __name__ == "__main__":

    test_prediction = "Tomato Target Spot"

    display_advisory(test_prediction)
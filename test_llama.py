import ollama

# Disease information from our verified database
disease_info = {
    "disease": "Tomato Target Spot",
    "crop": "Tomato",
    "precautions": [
        "Inspect affected plants regularly.",
        "Avoid unnecessary wetting of leaves.",
        "Consult local agricultural experts for treatment."
    ]
}

prompt = f"""
You are a farmer advisory assistant.

Explain the following disease information in simple English.

Disease: {disease_info['disease']}
Crop: {disease_info['crop']}

Precautions:
{chr(10).join(disease_info['precautions'])}

Rules:
1. Use only the information provided.
2. Do not invent medicines or treatments.
3. Give a short, clear explanation for farmers.
"""

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n===== ARM Bharat AI-SoC =====")
print("       FARMER ADVISORY")
print("=============================")
print(response["message"]["content"])
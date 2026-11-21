from huggingface_hub import hf_hub_download

print("Starting PlantVillage image download...")

file_path = hf_hub_download(
    repo_id="mohanty/PlantVillage",
    filename="data.zip",
    repo_type="dataset"
)

print("\nImage archive downloaded successfully!")
print("File location:", file_path)
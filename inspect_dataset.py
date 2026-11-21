from datasets import load_dataset

print("Downloading PlantVillage dataset...")

dataset = load_dataset("mohanty/PlantVillage", "default")

print("\nDataset downloaded successfully!")
print(dataset)

for split in dataset:
    print(split, ":", len(dataset[split]))

print("\nDataset features:")
print(dataset["train"].features)

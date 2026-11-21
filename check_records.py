from datasets import load_dataset

dataset = load_dataset("mohanty/PlantVillage", "default")

print("FIRST TRAINING RECORD:")
print(dataset["train"][0]["text"])

print("\nFIRST FIVE RECORDS:")
for i in range(5):
    print(i + 1, dataset["train"][i]["text"])
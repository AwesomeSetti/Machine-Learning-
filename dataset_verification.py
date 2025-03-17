import json
from collections import Counter
import os

# Define dataset paths
DATASET_PATH = "/users/setti/Desktop/ML/Assignment 1"  # Update to your dataset folder
train_path = os.path.join(DATASET_PATH, "training.json")
val_path = os.path.join(DATASET_PATH, "validation.json")
test_path = os.path.join(DATASET_PATH, "test.json")

# Load datasets
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: {file_path} not found!")
        return []

train_data = load_json(train_path)
val_data = load_json(val_path)
test_data = load_json(test_path)

# Count unique labels
def count_labels(data, name):
    labels = [int(entry["stars"]) for entry in data]
    print(f"🔹 {name} Data Label Distribution:", Counter(labels))

count_labels(train_data, "Training")
count_labels(val_data, "Validation")
count_labels(test_data, "Test")

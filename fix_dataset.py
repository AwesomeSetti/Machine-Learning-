import json
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split

DATASET_PATH = "/users/setti/Desktop/ML/Assignment 1"
fixed_dataset_path = "/users/setti/Desktop/ML/Assignment 1"

# Load datasets
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_json(os.path.join(DATASET_PATH, "training.json"))
val_data = load_json(os.path.join(DATASET_PATH, "validation.json"))
test_data = load_json(os.path.join(DATASET_PATH, "test.json"))

# Merge all datasets
full_dataset = train_data + val_data + test_data
random.shuffle(full_dataset)

# Resplit into new sets
train_data_fixed, temp_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
val_data_fixed, test_data_fixed = train_test_split(temp_data, test_size=0.5, random_state=42)

# Check label distribution
def count_labels(data, name):
    labels = [int(entry["stars"]) for entry in data]
    print(f"🔹 {name} Data Label Distribution:", Counter(labels))

count_labels(train_data_fixed, "Fixed Training")
count_labels(val_data_fixed, "Fixed Validation")
count_labels(test_data_fixed, "Fixed Test")

# Save the new datasets
os.makedirs(fixed_dataset_path, exist_ok=True)

with open(os.path.join(fixed_dataset_path, "training_fixed.json"), "w") as f:
    json.dump(train_data_fixed, f, indent=4)

with open(os.path.join(fixed_dataset_path, "validation_fixed.json"), "w") as f:
    json.dump(val_data_fixed, f, indent=4)

with open(os.path.join(fixed_dataset_path, "test_fixed.json"), "w") as f:
    json.dump(test_data_fixed, f, indent=4)

print("✅ Fixed datasets saved successfully!")

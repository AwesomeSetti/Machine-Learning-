import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
import os

# =======================
# 📌 Step 1: Load & Verify Dataset
# =======================

DATASET_PATH = "datasets_fixed/"  # Update this path if needed
train_path = os.path.join(DATASET_PATH, "training_fixed.json")
val_path = os.path.join(DATASET_PATH, "validation_fixed.json")
test_path = os.path.join(DATASET_PATH, "test_fixed.json")

# Load datasets
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_json(train_path)
val_data = load_json(val_path)
test_data = load_json(test_path)

# Verify label distribution
def count_labels(data, name):
    labels = [int(entry["stars"]) for entry in data]
    print(f"🔹 {name} Data Label Distribution:", Counter(labels))

count_labels(train_data, "Training")
count_labels(val_data, "Validation")
count_labels(test_data, "Test")

# =======================
# 📌 Step 2: Preprocess Text Data (Convert to Vectors)
# =======================

# Create vocabulary
def make_vocab(data):
    word_counts = Counter()
    for entry in data:
        words = entry["text"].lower().split()
        word_counts.update(words)

    word2index = {word: i for i, (word, _) in enumerate(word_counts.items())}
    word2index["<UNK>"] = len(word2index)
    return word2index

word2index = make_vocab(train_data)

# Convert text to Bag-of-Words representation
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    
    for entry in data:
        words = entry["text"].lower().split()
        vector = torch.zeros(len(word2index))

        for word in words:
            index = word2index.get(word, word2index["<UNK>"])
            vector[index] += 1  

        label = int(entry["stars"])  
        vectorized_data.append((vector, label))

    return vectorized_data

# Convert datasets into numerical vectors
train_data_vectorized = convert_to_vector_representation(train_data, word2index)
val_data_vectorized = convert_to_vector_representation(val_data, word2index)
test_data_vectorized = convert_to_vector_representation(test_data, word2index)

print("✅ Text data successfully converted into numerical format!")

# =======================
# 📌 Step 3: Define the FFNN Model
# =======================

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        inputs = inputs.view(1, -1)  # Ensure correct input shape
        hidden = self.relu(self.fc1(inputs))
        hidden = self.dropout(hidden)
        output = self.softmax(self.fc2(hidden))
        return output

# =======================
# 📌 Step 4: Train the Model (FIXED TRAINING FUNCTION)
# =======================

def train_ffnn(model, train_data, val_data, epochs=10, batch_size=32, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for input_vector, gold_label in tqdm(train_data):
            optimizer.zero_grad()
            outputs = model(input_vector)  # ✅ No reshaping needed

            loss = loss_function(outputs, torch.tensor([gold_label], dtype=torch.long))  # ✅ FIXED LABEL TYPE

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += int(torch.argmax(outputs) == gold_label)  # ✅ FIXED ACCURACY CALCULATION
            total += 1
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {correct / total:.4f}")

# Initialize and Train Model
input_dim = len(word2index)
hidden_dim = 128
output_dim = 5

ffnn_model = FFNN(input_dim, hidden_dim, output_dim)
train_ffnn(ffnn_model, train_data_vectorized, val_data_vectorized, epochs=10, batch_size=64, learning_rate=0.0005)

# =======================
# 📌 Step 5: Evaluate on Test Data (FIXED)
# =======================

def evaluate_ffnn_test(model, test_data_vectorized):
    """Evaluate FFNN model performance on test data."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for input_vector, gold_label in test_data_vectorized:
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)

            correct += int(predicted_label == gold_label)
            total += 1

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

# Run Test Evaluation
evaluate_ffnn_test(ffnn_model, test_data_vectorized)

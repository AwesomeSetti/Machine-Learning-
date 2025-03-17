import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

# Load dataset
train_path = "datasets_fixed/training_fixed.json"
val_path = "datasets_fixed/validation_fixed.json"
test_path = "datasets_fixed/test_fixed.json"

with open(train_path, "r") as f:
    train_data = json.load(f)

with open(val_path, "r") as f:
    val_data = json.load(f)

with open(test_path, "r") as f:
    test_data = json.load(f)

# Define Model
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Add dropout for better generalization
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        hidden = self.relu(self.fc1(inputs))
        hidden = self.dropout(hidden)
        output = self.softmax(self.fc2(hidden))
        return output

# Training Function
def train_ffnn(model, train_data, val_data, epochs=10, batch_size=32, learning_rate=0.001):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for input_vector, gold_label in tqdm(train_data):
            optimizer.zero_grad()
            outputs = model(input_vector)
            loss = loss_function(outputs, gold_label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (torch.argmax(outputs, dim=1) == gold_label).sum().item()
            total += gold_label.size(0)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {correct / total:.4f}")

# Train Model
input_dim = 5000  # Adjust based on vocab size
hidden_dim = 128
output_dim = 5

ffnn_model = FFNN(input_dim, hidden_dim, output_dim)
train_ffnn(ffnn_model, train_data, val_data, epochs=15, batch_size=64, learning_rate=0.0005)

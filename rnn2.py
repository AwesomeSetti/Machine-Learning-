import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
from tqdm import tqdm
import json
import pickle

# Load Dataset
with open("training.json", "r") as f:
    train_data = json.load(f)

with open("validation.json", "r") as f:
    val_data = json.load(f)

with open("test.json", "r") as f:
    test_data = json.load(f)

# Load Word Embeddings
with open("word_embedding.pkl", "rb") as f:
    word_embedding = pickle.load(f)

# Convert text into word embeddings
def process_text(text, word_embedding):
    text = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
    vectors = [word_embedding[word] if word in word_embedding else word_embedding["unk"] for word in text]
    return torch.tensor(vectors).view(len(vectors), 1, -1)

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        output_seq, hidden = self.rnn(inputs)
        summed_scores = torch.sum(self.fc(output_seq), dim=0)  
        predicted_vector = self.softmax(summed_scores)
        return predicted_vector

    def compute_loss(self, predictions, targets):
        return nn.NLLLoss()(predictions, targets)

# Training Function (Fixed to Print Loss)
def train_rnn(model, train_data, val_data, word_embedding, epochs=10, batch_size=16, learning_rate=0.001):
    """Train the RNN model with word embeddings"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loss_count = 0
        correct, total = 0, 0

        # Shuffle data for each epoch
        random.shuffle(train_data)

        for i in tqdm(range(0, len(train_data), batch_size)):  
            batch = train_data[i:i+batch_size]
            
            optimizer.zero_grad()
            loss = None
            
            for entry in batch:
                input_tensor = process_text(entry["text"], word_embedding).to(device)
                gold_label = torch.tensor([int(entry["stars"]) - 1], dtype=torch.long).to(device)

                output = model(input_tensor)
                example_loss = loss_function(output.view(1, -1), gold_label)

                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            # Normalize loss and backpropagate
            loss = loss / batch_size
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_count += 1

            # 🔹 Print loss every 10 minibatches
            if (loss_count % 10 == 0):
                print(f"Epoch {epoch+1}, Minibatch {loss_count}: Loss = {loss.item():.4f}")

        train_accuracy = correct / total
        val_accuracy = evaluate_rnn(model, val_data, word_embedding)

        # 🔹 Print average loss at the end of each epoch
        avg_loss = total_loss / loss_count
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

# Validation Function
def evaluate_rnn(model, val_data, word_embedding):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for entry in val_data:
            input_tensor = process_text(entry["text"], word_embedding)
            gold_label = torch.tensor([int(entry["stars"]) - 1], dtype=torch.long)

            output = model(input_tensor)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

    return correct / total

# Run Training
input_dim = 50  # Embedding size
hidden_dim = 64
output_dim = 5  # Five-star rating classification

rnn_model = RNN(input_dim, hidden_dim, output_dim)

train_rnn(rnn_model, train_data, val_data, word_embedding, epochs=10, batch_size=16, learning_rate=0.001)

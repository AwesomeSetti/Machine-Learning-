
import string
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

def build_vocab(data):
    """Build vocabulary from training data"""
    word_counts = Counter()
    for entry in data:
        words = entry["text"].lower().split()
        word_counts.update(words)
    return {word: i for i, (word, _) in enumerate(word_counts.items())}

def convert_to_bow(data, vocab):
    """Convert text data to Bag-of-Words vectors"""
    bow_vectors = []
    labels = []

    for entry in data:
        words = entry["text"].lower().split()
        bow_vector = [0] * len(vocab)

        for word in words:
            if word in vocab:
                bow_vector[vocab[word]] += 1  

        bow_vectors.append(bow_vector)
        labels.append(int(entry["stars"]) - 1)

    return bow_vectors, labels

# Load Word Embeddings
with open("word_embedding.pkl", "rb") as f:
    word_embedding = pickle.load(f)

def process_text(text, word_embedding):
    """Convert text into sequence of word embeddings."""
    text = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
    vectors = [word_embedding[word] if word in word_embedding else word_embedding["unk"] for word in text]
    return torch.tensor(vectors).view(len(vectors), 1, -1)

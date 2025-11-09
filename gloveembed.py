"""
gloveembed.py
------------------
Utility for loading pre-trained GloVe word embeddings and
constructing an embedding matrix aligned with the project vocabulary.

"""

import numpy as np
import torch
from tqdm import tqdm


def load_glove_vectors(glove_path: str) -> dict:
    
    word_to_vec = {}

    print(f"Loading GloVe embeddings from: {glove_path}")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading GloVe file", total=None):
            parts = line.strip().split()
            if len(parts) < 10:
                # Skip malformed lines
                continue
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            word_to_vec[word] = vector

    print(f"Loaded {len(word_to_vec):,} word vectors from {glove_path}")
    return word_to_vec


def build_embedding_matrix(tokens2index: dict, glove_path: str, embed_dim: int = 200) -> torch.FloatTensor:
    
    word_to_vec = load_glove_vectors(glove_path)
    vocab_size = len(tokens2index)

    # Random initialization for unknown words
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim)).astype(np.float32)

    found = 0
    for word, idx in tokens2index.items():
        vector = word_to_vec.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            found += 1

    print(f"Found {found:,}/{vocab_size:,} words in GloVe ({found / vocab_size:.2%} coverage).")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    return torch.tensor(embedding_matrix)

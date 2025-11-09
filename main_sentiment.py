import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from LSTM import LSTMModel
from gloveembed import build_embedding_matrix
import json
import numpy as np
import time
import os



# Saving checkpoints

def save_checkpoint(ckp_path, model, epoch, step, optimizer):
    checkpoint = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, ckp_path)
    print(f"Checkpoint saved to {ckp_path}")



# Main training + evaluation loop

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Configurations
 
    mode = "train"              # 'train' or 'test'
    batch_size = 300
    n_layers = 1
    input_len = 150
    embedding_dim = 200          # matches the GloVe file (200 or 300)
    hidden_dim = 128
    output_size = 1
    num_epochs = 5               
    learning_rate = 0.002
    clip = 5                     # gradient clipping
    pretrain = True              # enable GloVe
    glove_file = "glove/glove.6B.200d.txt"
    ckp_path = "checkpoints/sentiment_200d.pt"
    load_checkpoint = False      # set True to resume training

    os.makedirs("checkpoints", exist_ok=True)

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_dataset = MovieDataset("training_data.csv")
    test_dataset = MovieDataset("test_data.csv")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # ----------------------------
    # Load vocab + embeddings
    # ----------------------------
    with open("tokens2index.json", "r") as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print("Loading pre-trained GloVe embeddings...")
        embedding_matrix = build_embedding_matrix(tokens2index, glove_file, embed_dim=embedding_dim)
    else:
        embedding_matrix = None

    # ----------------------------
    # Initialize model
    # ----------------------------
    model = LSTMModel(
        vocab_size=vocab_size,
        output_size=output_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        input_len=input_len,
        pretrain=pretrain
    ).to(device)

    print(f"Model initialized with vocab size: {vocab_size}, hidden dim: {hidden_dim}")

    # ----------------------------
    # Define optimizer + loss
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # ----------------------------
    # Load checkpoint
    # ----------------------------
    start_epoch = 0
    if load_checkpoint and os.path.exists(ckp_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(ckp_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    # ----------------------------
    # Training loop
    # ----------------------------
    if mode == "train":
        print("\n-----------Start Training Now-----------\n")
        model.train()

        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            for x_batch, y_labels in train_loader:

                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                # Forward
                y_out = model(x_batch)
                loss = loss_fn(y_out, y_labels)

                # Back Propagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip) # prevents exploding gradient
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            save_checkpoint(ckp_path, model, epoch, 0, optimizer)

        print("\n-------------Training Completed-------------\n")

    # ----------------------------
    # Evaluation loop
    # ----------------------------
    print("Evaluating model on test set...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_batch, y_labels in test_loader:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            preds = torch.round(y_out)

            y_true.extend(y_labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # Compare performance of different embeddings
    # --------------------------------------------------
    print(f"Embedding Dimension Used: {embedding_dim}D")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"\nTotal runtime: {(time_end - time_start)/60:.2f} minutes")

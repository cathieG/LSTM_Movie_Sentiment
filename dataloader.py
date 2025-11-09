import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ast import literal_eval


class SentimentDataset(Dataset):
    

    def __init__(self, csv_file):
        # Read the preprocessed CSV and safely parse lists in 'input_x'
        self.data = pd.read_csv(csv_file, converters={'input_x': literal_eval})

    def __len__(self):
        # Total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a single sample at index `idx`
        row = self.data.iloc[idx]

        # Convert review (list of token IDs) and label to tensors
        input_x = torch.tensor(row['input_x'], dtype=torch.long)
        label = torch.tensor(row['Label'], dtype=torch.float)

        return input_x, label


def get_dataloaders(train_csv, test_csv, batch_size=64, num_workers=0):
    
    train_dataset = SentimentDataset(train_csv)
    test_dataset = SentimentDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader



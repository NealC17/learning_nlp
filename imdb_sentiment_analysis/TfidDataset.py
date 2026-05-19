import torch
from torch.utils.data import Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TfidDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].toarray().squeeze(0)
        x = torch.tensor(x, dtype=torch.float).to(device)

        label = torch.tensor(self.y.iloc[idx], dtype=torch.float).to(device)
        return x, label
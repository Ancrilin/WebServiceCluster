from torch.utils.data import Dataset
import torch
import numpy as np


class myDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = np.array(dataset)
        self.config = config

    def __getitem__(self, index: int):
        xi, xj, xk = self.dataset[index]
        return (torch.tensor(xi, dtype=torch.long).to(self.config.device),
                torch.tensor(xj, dtype=torch.long).to(self.config.device),
                torch.tensor(xk, dtype=torch.long).to(self.config.device)
                )

    def __len__(self) -> int:
        return len(self.dataset)
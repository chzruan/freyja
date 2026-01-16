"""
Dataset class for halo beta(r) emulator training.
Standard PyTorch dataset.
"""

import torch
from torch.utils.data import Dataset


class HaloBetaDataset(Dataset):
    def __init__(self, inputs, targets, weights):
        """
        inputs: (N_samples, 6) -> [cosmo..., u, v]
        targets: (N_samples, N_r_cut) -> beta(r)
        weights: (N_samples, N_r_cut) -> 1 / sigma_beta^2
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.weights[idx]

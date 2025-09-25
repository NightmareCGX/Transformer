import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ByteDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"not found: {data_path}")
        data = np.load(data_path, mmap_mode="r")
        self.data = torch.from_numpy(data).long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]

        return x, y


def get_dataloader(cfg):
    ds = cfg.get("dataset", {})
    train_path = ds.get("train_path")
    val_path = ds.get("val_path")
    if train_path is None or val_path is None:
        raise KeyError("dataset paths not found")

    seq_len = int(cfg["model"].get("seq_len"))
    batch_size = int(cfg["model"].get("batch_size"))
    seed = cfg.get("seed")

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))

    train_dataset = ByteDataset(train_path, seq_len)
    val_dataset = ByteDataset(val_path, seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )

    return train_loader, val_loader

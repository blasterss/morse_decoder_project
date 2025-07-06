import pandas as pd
import numpy as np

import json
from typing import Dict, Literal

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MorseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 split: str = 'train', 
                 val_size: int = 1000, 
                 random_seed: int = 42):
        
        self.df = df
        self.df.dropna(subset=["morse_feats", "morse_text"], inplace=True)

        self.data = []
        for _, row in self.df.iterrows():
            seq = np.array(json.loads(row['morse_feats']), dtype=np.float32)
            target = np.array(json.loads(row['morse_text']), dtype=np.int64)
            if len(seq) > 0 and len(target) > 0:
                self.data.append((seq, target))

        total = len(self.data)
        train_size = total - val_size
        
        generator = torch.Generator().manual_seed(random_seed)
        datasets = random_split(self.data, [train_size, val_size], generator=generator)

        if split == 'train':
            self.data = datasets[0]
        elif split == 'val':
            self.data = datasets[1]
        elif split == 'all':
            self.data = self.data
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)

def _collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = [len(x) for x in inputs]
    target_lengths = [len(y) for y in targets]

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_concat = torch.cat(targets)

    return inputs_padded, targets_concat, input_lengths, target_lengths

def get_dataloaders(
    df: pd.DataFrame,
    val_size: int,
    random_seed: int,
    batch_size: int
) -> Dict[Literal["train", "validation"], DataLoader]:
    """
    Создает и возвращает словарь с DataLoader'ами для обучения и валидации
    
    Args:
        df: Датафрейм
        val_size: Размер валидационной выборки
        random_seed: Сид для воспроизводимости
        batch_size: Размер батча
        
    Returns:
        Словарь с DataLoader'ами для обучения и валидации
    """
    train_dataset = MorseDataset(
        df,
        split='train',
        val_size=val_size,
        random_seed=random_seed
    )
    
    val_dataset = MorseDataset(
        df,
        split='val',
        val_size=val_size,
        random_seed=random_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn
    )

    dataloaders = {
        "train": train_dataloader,
        "validation": val_dataloader
    }

    return dataloaders


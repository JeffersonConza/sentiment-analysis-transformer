"""
Dataset handling and DataLoader creation for IMDB reviews.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def load_text_files(folder):
    """
    Reads all .txt files in a directory and returns their content as a list of strings.
    """
    reviews = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
        
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                reviews.append(f.read())
    return reviews


def build_imdb_dataframes(root_dir="aclImdb", val_split=0.1, random_state=42):
    """
    Loads IMDB review text files from raw directory structure and creates
    train, validation, and test pandas DataFrames.
    """
    train_pos = load_text_files(os.path.join(root_dir, "train", "pos"))
    train_neg = load_text_files(os.path.join(root_dir, "train", "neg"))
    test_pos = load_text_files(os.path.join(root_dir, "test", "pos"))
    test_neg = load_text_files(os.path.join(root_dir, "test", "neg"))

    full_train_df = pd.DataFrame({
        "review": train_pos + train_neg,
        "label": [1] * len(train_pos) + [0] * len(train_neg)
    })

    test_df = pd.DataFrame({
        "review": test_pos + test_neg,
        "label": [1] * len(test_pos) + [0] * len(test_neg)
    })

    # Shuffle and split train into train / val
    shuffled_df = full_train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_size = int((1.0 - val_split) * len(shuffled_df))
    
    train_df = shuffled_df.iloc[:train_size].reset_index(drop=True)
    val_df = shuffled_df.iloc[train_size:].reset_index(drop=True)

    return train_df, val_df, test_df


class IMDBDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized IMDB review texts and labels.
    """

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["review"])
        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        return input_ids, label


def get_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=32, max_length=128):
    """
    Utility function to instantiate PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = IMDBDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = IMDBDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = IMDBDataset(test_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

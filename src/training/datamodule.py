import torch
import pytorch_lightning as lightning
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ClassificationPrefixDataset(Dataset):
    """Dataset for classification tasks (next_activity)."""
    def __init__(self, prefixes_dataframe, label_series, vocabulary):
        self.prefixes_dataframe = prefixes_dataframe
        self.label_series = label_series
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.prefixes_dataframe)

    def __getitem__(self, index):
        encoded_sequence = self.vocabulary.encode_sequence(
            self.prefixes_dataframe.iloc[index]["prefix_activities"]
        )
        target_index = self.vocabulary.token_to_index[self.label_series.iloc[index]]
        return torch.tensor(encoded_sequence, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)


class RegressionPrefixDataset(Dataset):
    """Dataset for regression tasks (next_time, remaining_time)."""
    def __init__(self, prefixes_dataframe, label_series, vocabulary):
        self.prefixes_dataframe = prefixes_dataframe
        self.label_series = label_series
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.prefixes_dataframe)

    def __getitem__(self, index):
        encoded_sequence = self.vocabulary.encode_sequence(
            self.prefixes_dataframe.iloc[index]["prefix_activities"]
        )
        target_value = float(self.label_series.iloc[index])
        return torch.tensor(encoded_sequence, dtype=torch.long), torch.tensor(target_value, dtype=torch.float)


class SuffixPrefixDataset(Dataset):
    """Dataset for suffix prediction tasks."""
    def __init__(self, prefixes_dataframe, label_series, vocabulary):
        self.prefixes_dataframe = prefixes_dataframe
        self.label_series = label_series
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.prefixes_dataframe)

    def __getitem__(self, index):
        encoded_sequence = self.vocabulary.encode_sequence(
            self.prefixes_dataframe.iloc[index]["prefix_activities"]
        )
        # For suffix, we need to encode the suffix sequence
        suffix_sequence = self.label_series.iloc[index]
        if isinstance(suffix_sequence, list):
            # Convert string tokens to indices
            encoded_suffix = []
            for token in suffix_sequence:
                if token in self.vocabulary.token_to_index:
                    encoded_suffix.append(self.vocabulary.token_to_index[token])
                else:
                    encoded_suffix.append(0)  # Pad token
        else:
            encoded_suffix = [0] * 10  # Default padding
        
        return torch.tensor(encoded_sequence, dtype=torch.long), torch.tensor(encoded_suffix, dtype=torch.long)


def pad_and_collate(batch, pad_index=0):
    sequences, targets = zip(*batch)
    max_length = max(sequence.size(0) for sequence in sequences)
    padded_sequences = torch.full((len(sequences), max_length), pad_index, dtype=torch.long)
    for batch_index, sequence in enumerate(sequences):
        padded_sequences[batch_index, :sequence.size(0)] = sequence
    return padded_sequences, torch.stack(targets)


def pad_and_collate_suffix(batch, pad_index=0):
    sequences, targets = zip(*batch)
    max_seq_length = max(sequence.size(0) for sequence in sequences)
    max_target_length = max(target.size(0) for target in targets)
    
    padded_sequences = torch.full((len(sequences), max_seq_length), pad_index, dtype=torch.long)
    padded_targets = torch.full((len(sequences), max_target_length), pad_index, dtype=torch.long)
    
    for batch_index, (sequence, target) in enumerate(zip(sequences, targets)):
        padded_sequences[batch_index, :sequence.size(0)] = sequence
        padded_targets[batch_index, :target.size(0)] = target
    
    return padded_sequences, padded_targets


class ClassificationPrefixDataModule(lightning.LightningDataModule):
    """DataModule for classification tasks (next_activity)."""
    def __init__(self, train_prefix_df, train_labels, val_prefix_df, val_labels, vocabulary, batch_size=128):
        super().__init__()
        self.train_dataset = ClassificationPrefixDataset(train_prefix_df, train_labels, vocabulary)
        self.val_dataset = ClassificationPrefixDataset(val_prefix_df, val_labels, vocabulary)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_and_collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate, num_workers=0)


class RegressionPrefixDataModule(lightning.LightningDataModule):
    """DataModule for regression tasks (next_time, remaining_time)."""
    def __init__(self, train_prefix_df, train_labels, val_prefix_df, val_labels, vocabulary, batch_size=128):
        super().__init__()
        self.train_dataset = RegressionPrefixDataset(train_prefix_df, train_labels, vocabulary)
        self.val_dataset = RegressionPrefixDataset(val_prefix_df, val_labels, vocabulary)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_and_collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate, num_workers=0)


class SuffixPrefixDataModule(lightning.LightningDataModule):
    """DataModule for suffix prediction tasks."""
    def __init__(self, train_prefix_df, train_labels, val_prefix_df, val_labels, vocabulary, batch_size=128):
        super().__init__()
        self.train_dataset = SuffixPrefixDataset(train_prefix_df, train_labels, vocabulary)
        self.val_dataset = SuffixPrefixDataset(val_prefix_df, val_labels, vocabulary)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_and_collate_suffix, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate_suffix, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate_suffix, num_workers=0)




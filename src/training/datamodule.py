import torch
import pytorch_lightning as lightning
from torch.utils.data import Dataset, DataLoader

class PrefixDataset(Dataset):
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

def pad_and_collate(batch, pad_index=0):
    sequences, targets = zip(*batch)
    max_length = max(sequence.size(0) for sequence in sequences)
    padded_sequences = torch.full((len(sequences), max_length), pad_index, dtype=torch.long)
    for batch_index, sequence in enumerate(sequences):
        padded_sequences[batch_index, :sequence.size(0)] = sequence
    return padded_sequences, torch.stack(targets)

class PrefixDataModule(lightning.LightningDataModule):
    def __init__(self, train_prefix_df, train_labels, val_prefix_df, val_labels, vocabulary, batch_size=128):
        super().__init__()
        self.train_dataset = PrefixDataset(train_prefix_df, train_labels, vocabulary)
        self.val_dataset = PrefixDataset(val_prefix_df, val_labels, vocabulary)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_and_collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_and_collate, num_workers=0)

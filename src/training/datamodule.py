"""
Data modules for PyTorch Lightning training.
Supports the ProcessTransformer data format with adapter compatibility.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import pytorch_lightning as lightning

from ..data.loader import LogsDataLoader


class NextActivityDataset(Dataset):
    """Dataset for next activity prediction using ProcessTransformer format."""
    
    def __init__(self, df: pd.DataFrame, x_word_dict: Dict[str, int], 
                 y_word_dict: Dict[str, int], max_case_length: int):
        self.df = df
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.max_case_length = max_case_length
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare tokenized sequences and labels."""
        x = self.df["prefix"].values
        y = self.df["next_act"].values
        
        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.x_word_dict[s] for s in _x.split()])
        
        # Pad sequences
        token_x = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in token_x],
            batch_first=True,
            padding_value=0
        )
        
        # Truncate or pad to max_case_length
        if token_x.size(1) > self.max_case_length:
            token_x = token_x[:, :self.max_case_length]
        elif token_x.size(1) < self.max_case_length:
            padding = torch.zeros(token_x.size(0), self.max_case_length - token_x.size(1), dtype=torch.long)
            token_x = torch.cat([token_x, padding], dim=1)
        
        # Tokenize labels
        token_y = torch.tensor([self.y_word_dict[_y] for _y in y], dtype=torch.long)
        
        self.token_x = token_x
        self.token_y = token_y
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.token_x[idx], self.token_y[idx]


class NextActivityDataModule(lightning.LightningDataModule):
    """Data module for next activity prediction using ProcessTransformer format."""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 x_word_dict: Dict[str, int], y_word_dict: Dict[str, int],
                 max_case_length: int, batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.max_case_length = max_case_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Split train_df into train and validation
        train_size = int(0.8 * len(train_df))
        self.train_subset = train_df.iloc[:train_size]
        self.val_subset = train_df.iloc[train_size:]
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            self.train_dataset = NextActivityDataset(
                self.train_subset, self.x_word_dict, self.y_word_dict, self.max_case_length
            )
            self.val_dataset = NextActivityDataset(
                self.val_subset, self.x_word_dict, self.y_word_dict, self.max_case_length
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = NextActivityDataset(
                self.test_df, self.x_word_dict, self.y_word_dict, self.max_case_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class NextTimeDataset(Dataset):
    """Dataset for next time prediction using ProcessTransformer format."""
    
    def __init__(self, df: pd.DataFrame, x_word_dict: Dict[str, int], 
                 max_case_length: int, time_scaler=None, y_scaler=None):
        self.df = df
        self.x_word_dict = x_word_dict
        self.max_case_length = max_case_length
        self.time_scaler = time_scaler
        self.y_scaler = y_scaler
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare tokenized sequences, time features, and labels."""
        x = self.df["prefix"].values
        time_x = self.df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = self.df["next_time"].values.astype(np.float32)
        
        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.x_word_dict[s] for s in _x.split()])
        
        # Pad sequences
        token_x = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in token_x],
            batch_first=True,
            padding_value=0
        )
        
        # Truncate or pad to max_case_length
        if token_x.size(1) > self.max_case_length:
            token_x = token_x[:, :self.max_case_length]
        elif token_x.size(1) < self.max_case_length:
            padding = torch.zeros(token_x.size(0), self.max_case_length - token_x.size(1), dtype=torch.long)
            token_x = torch.cat([token_x, padding], dim=1)
        
        # Scale time features and labels
        if self.time_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.time_scaler = StandardScaler()
            time_x = self.time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = self.time_scaler.transform(time_x).astype(np.float32)
        
        if self.y_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.y_scaler = StandardScaler()
            y = self.y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = self.y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.token_x = torch.tensor(token_x, dtype=torch.long)
        self.time_x = torch.tensor(time_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.token_x[idx], self.time_x[idx], self.y[idx]


class NextTimeDataModule(lightning.LightningDataModule):
    """Data module for next time prediction using ProcessTransformer format."""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 x_word_dict: Dict[str, int], y_word_dict: Dict[str, int],
                 max_case_length: int, batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.max_case_length = max_case_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Split train_df into train and validation
        train_size = int(0.8 * len(train_df))
        self.train_subset = train_df.iloc[:train_size]
        self.val_subset = train_df.iloc[train_size:]
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            self.train_dataset = NextTimeDataset(
                self.train_subset, self.x_word_dict, self.max_case_length
            )
            self.val_dataset = NextTimeDataset(
                self.val_subset, self.x_word_dict, self.max_case_length,
                time_scaler=self.train_dataset.time_scaler,
                y_scaler=self.train_dataset.y_scaler
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = NextTimeDataset(
                self.test_df, self.x_word_dict, self.max_case_length,
                time_scaler=self.train_dataset.time_scaler,
                y_scaler=self.train_dataset.y_scaler
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class RemainingTimeDataset(Dataset):
    """Dataset for remaining time prediction using ProcessTransformer format."""
    
    def __init__(self, df: pd.DataFrame, x_word_dict: Dict[str, int], 
                 max_case_length: int, time_scaler=None, y_scaler=None):
        self.df = df
        self.x_word_dict = x_word_dict
        self.max_case_length = max_case_length
        self.time_scaler = time_scaler
        self.y_scaler = y_scaler
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare tokenized sequences, time features, and labels."""
        x = self.df["prefix"].values
        time_x = self.df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = self.df["remaining_time_days"].values.astype(np.float32)
        
        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.x_word_dict[s] for s in _x.split()])
        
        # Pad sequences
        token_x = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in token_x],
            batch_first=True,
            padding_value=0
        )
        
        # Truncate or pad to max_case_length
        if token_x.size(1) > self.max_case_length:
            token_x = token_x[:, :self.max_case_length]
        elif token_x.size(1) < self.max_case_length:
            padding = torch.zeros(token_x.size(0), self.max_case_length - token_x.size(1), dtype=torch.long)
            token_x = torch.cat([token_x, padding], dim=1)
        
        # Scale time features and labels
        if self.time_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.time_scaler = StandardScaler()
            time_x = self.time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = self.time_scaler.transform(time_x).astype(np.float32)
        
        if self.y_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.y_scaler = StandardScaler()
            y = self.y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = self.y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.token_x = torch.tensor(token_x, dtype=torch.long)
        self.time_x = torch.tensor(time_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.token_x[idx], self.time_x[idx], self.y[idx]


class RemainingTimeDataModule(lightning.LightningDataModule):
    """Data module for remaining time prediction using ProcessTransformer format."""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 x_word_dict: Dict[str, int], y_word_dict: Dict[str, int],
                 max_case_length: int, batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.max_case_length = max_case_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Split train_df into train and validation
        train_size = int(0.8 * len(train_df))
        self.train_subset = train_df.iloc[:train_size]
        self.val_subset = train_df.iloc[train_size:]
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            self.train_dataset = RemainingTimeDataset(
                self.train_subset, self.x_word_dict, self.max_case_length
            )
            self.val_dataset = RemainingTimeDataset(
                self.val_subset, self.x_word_dict, self.max_case_length,
                time_scaler=self.train_dataset.time_scaler,
                y_scaler=self.train_dataset.y_scaler
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = RemainingTimeDataset(
                self.test_df, self.x_word_dict, self.max_case_length,
                time_scaler=self.train_dataset.time_scaler,
                y_scaler=self.train_dataset.y_scaler
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class ClassificationPrefixDataModule(lightning.LightningDataModule):
    """Legacy data module for backward compatibility."""
    
    def __init__(self, train_prefix_df: pd.DataFrame, train_labels: pd.Series,
                 val_prefix_df: pd.DataFrame, val_labels: pd.Series,
                 vocabulary, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.train_prefix_df = train_prefix_df
        self.train_labels = train_labels
        self.val_prefix_df = val_prefix_df
        self.val_labels = val_labels
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Convert to tensors
            self.train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.train_prefix_df.values, dtype=torch.long),
                torch.tensor(self.train_labels.values, dtype=torch.long)
            )
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.val_prefix_df.values, dtype=torch.long),
                torch.tensor(self.val_labels.values, dtype=torch.long)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class RegressionPrefixDataModule(lightning.LightningDataModule):
    """Data module for regression tasks (next time, remaining time)."""
    
    def __init__(self, train_prefix_df: pd.DataFrame, train_labels: pd.Series,
                 val_prefix_df: pd.DataFrame, val_labels: pd.Series,
                 vocabulary, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.train_prefix_df = train_prefix_df
        self.train_labels = train_labels
        self.val_prefix_df = val_prefix_df
        self.val_labels = val_labels
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Convert to tensors
            self.train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.train_prefix_df.values, dtype=torch.long),
                torch.tensor(self.train_labels.values, dtype=torch.float32)
            )
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.val_prefix_df.values, dtype=torch.long),
                torch.tensor(self.val_labels.values, dtype=torch.float32)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class SuffixPrefixDataModule(lightning.LightningDataModule):
    """Data module for suffix prediction task."""
    
    def __init__(self, train_prefix_df: pd.DataFrame, train_labels: pd.Series,
                 val_prefix_df: pd.DataFrame, val_labels: pd.Series,
                 vocabulary, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.train_prefix_df = train_prefix_df
        self.train_labels = train_labels
        self.val_prefix_df = val_prefix_df
        self.val_labels = val_labels
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Convert to tensors
            self.train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.train_prefix_df.values, dtype=torch.long),
                torch.tensor(self.train_labels.values, dtype=torch.long)
            )
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.val_prefix_df.values, dtype=torch.long),
                torch.tensor(self.val_labels.values, dtype=torch.long)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )







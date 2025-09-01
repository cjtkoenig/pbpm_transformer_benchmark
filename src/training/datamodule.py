"""
Data modules for PyTorch Lightning training.
Supports the canonical 5-fold case-based cross-validation format.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import pytorch_lightning as lightning

from ..data.loader import CanonicalLogsDataLoader


class CanonicalNextActivityDataset(Dataset):
    """Dataset for next activity prediction using canonical format."""
    
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


class CanonicalNextTimeDataset(Dataset):
    """Dataset for next time prediction using canonical format."""
    
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
        
        # Scale time features
        if self.time_scaler is not None:
            time_x = self.time_scaler.transform(time_x).astype(np.float32)
        
        # Scale target
        if self.y_scaler is not None:
            y = self.y_scaler.transform(y.reshape(-1, 1)).astype(np.float32).flatten()
        
        self.token_x = token_x
        self.time_x = torch.tensor(time_x, dtype=torch.float32)
        self.token_y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.token_x[idx], self.time_x[idx], self.token_y[idx]


class CanonicalRemainingTimeDataset(Dataset):
    """Dataset for remaining time prediction using canonical format."""
    
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
        
        # Scale time features
        if self.time_scaler is not None:
            time_x = self.time_scaler.transform(time_x).astype(np.float32)
        
        # Scale target
        if self.y_scaler is not None:
            y = self.y_scaler.transform(y.reshape(-1, 1)).astype(np.float32).flatten()
        
        self.token_x = token_x
        self.time_x = torch.tensor(time_x, dtype=torch.float32)
        self.token_y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.token_x[idx], self.time_x[idx], self.token_y[idx]


class CanonicalNextActivityDataModule(lightning.LightningDataModule):
    """Data module for next activity prediction using canonical format."""
    
    def __init__(self, dataset_name: str, task: str, fold_idx: int,
                 processed_dir: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.dataset_name = dataset_name
        self.task = task
        self.fold_idx = fold_idx
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize canonical loader
        self.data_loader = CanonicalLogsDataLoader(dataset_name, processed_dir)
        
        # Load fold data
        self.train_df, self.val_df = self.data_loader.load_fold_data(task, fold_idx)
        
        # Calculate max case length
        self.max_case_length = self.data_loader.get_max_case_length(self.train_df)
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = CanonicalNextActivityDataset(
                self.train_df, self.data_loader.x_word_dict, 
                self.data_loader.y_word_dict, self.max_case_length
            )
            self.val_dataset = CanonicalNextActivityDataset(
                self.val_df, self.data_loader.x_word_dict, 
                self.data_loader.y_word_dict, self.max_case_length
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


class CanonicalNextTimeDataModule(lightning.LightningDataModule):
    """Data module for next time prediction using canonical format."""
    
    def __init__(self, dataset_name: str, task: str, fold_idx: int,
                 processed_dir: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.dataset_name = dataset_name
        self.task = task
        self.fold_idx = fold_idx
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize canonical loader
        self.data_loader = CanonicalLogsDataLoader(dataset_name, processed_dir)
        
        # Load fold data
        self.train_df, self.val_df = self.data_loader.load_fold_data(task, fold_idx)
        
        # Calculate max case length
        self.max_case_length = self.data_loader.get_max_case_length(self.train_df)
        
        # Initialize scalers
        self.time_scaler = None
        self.y_scaler = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Load persisted scalers if available; otherwise fit and persist
            import pickle
            from pathlib import Path as _Path
            fold_dir = _Path(self.processed_dir) / self.dataset_name / "splits" / self.task / f"fold_{self.fold_idx}"
            time_scaler_path = fold_dir / f"time_scaler_{self.task}.pkl"
            y_scaler_path = fold_dir / f"y_scaler_{self.task}.pkl"

            if time_scaler_path.exists() and y_scaler_path.exists():
                with open(time_scaler_path, "rb") as f:
                    self.time_scaler = pickle.load(f)
                with open(y_scaler_path, "rb") as f:
                    self.y_scaler = pickle.load(f)
            else:
                from sklearn.preprocessing import StandardScaler
                train_time_x = self.train_df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
                train_y = self.train_df["next_time"].values.astype(np.float32)
                self.time_scaler = StandardScaler().fit(train_time_x)
                self.y_scaler = StandardScaler().fit(train_y.reshape(-1, 1))
                # Persist for reproducibility
                fold_dir.mkdir(parents=True, exist_ok=True)
                with open(time_scaler_path, "wb") as f:
                    pickle.dump(self.time_scaler, f)
                with open(y_scaler_path, "wb") as f:
                    pickle.dump(self.y_scaler, f)
            
            self.train_dataset = CanonicalNextTimeDataset(
                self.train_df, self.data_loader.x_word_dict, 
                self.max_case_length, self.time_scaler, self.y_scaler
            )
            self.val_dataset = CanonicalNextTimeDataset(
                self.val_df, self.data_loader.x_word_dict, 
                self.max_case_length, self.time_scaler, self.y_scaler
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


class CanonicalRemainingTimeDataModule(lightning.LightningDataModule):
    """Data module for remaining time prediction using canonical format."""
    
    def __init__(self, dataset_name: str, task: str, fold_idx: int,
                 processed_dir: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.dataset_name = dataset_name
        self.task = task
        self.fold_idx = fold_idx
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize canonical loader
        self.data_loader = CanonicalLogsDataLoader(dataset_name, processed_dir)
        
        # Load fold data
        self.train_df, self.val_df = self.data_loader.load_fold_data(task, fold_idx)
        
        # Calculate max case length
        self.max_case_length = self.data_loader.get_max_case_length(self.train_df)
        
        # Initialize scalers
        self.time_scaler = None
        self.y_scaler = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Load persisted scalers if available; otherwise fit and persist
            import pickle
            from pathlib import Path as _Path
            fold_dir = _Path(self.processed_dir) / self.dataset_name / "splits" / self.task / f"fold_{self.fold_idx}"
            time_scaler_path = fold_dir / f"time_scaler_{self.task}.pkl"
            y_scaler_path = fold_dir / f"y_scaler_{self.task}.pkl"

            if time_scaler_path.exists() and y_scaler_path.exists():
                with open(time_scaler_path, "rb") as f:
                    self.time_scaler = pickle.load(f)
                with open(y_scaler_path, "rb") as f:
                    self.y_scaler = pickle.load(f)
            else:
                from sklearn.preprocessing import StandardScaler
                train_time_x = self.train_df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
                train_y = self.train_df["remaining_time_days"].values.astype(np.float32)
                self.time_scaler = StandardScaler().fit(train_time_x)
                self.y_scaler = StandardScaler().fit(train_y.reshape(-1, 1))
                # Persist for reproducibility
                fold_dir.mkdir(parents=True, exist_ok=True)
                with open(time_scaler_path, "wb") as f:
                    pickle.dump(self.time_scaler, f)
                with open(y_scaler_path, "wb") as f:
                    pickle.dump(self.y_scaler, f)
            
            self.train_dataset = CanonicalRemainingTimeDataset(
                self.train_df, self.data_loader.x_word_dict, 
                self.max_case_length, self.time_scaler, self.y_scaler
            )
            self.val_dataset = CanonicalRemainingTimeDataset(
                self.val_df, self.data_loader.x_word_dict, 
                self.max_case_length, self.time_scaler, self.y_scaler
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


# Legacy data modules for backward compatibility (deprecated)
class NextActivityDataset(Dataset):
    """Legacy dataset for next activity prediction (deprecated)."""
    
    def __init__(self, df: pd.DataFrame, x_word_dict: Dict[str, int], 
                 y_word_dict: Dict[str, int], max_case_length: int):
        import warnings
        warnings.warn("NextActivityDataset is deprecated. Use CanonicalNextActivityDataset instead.", 
                     DeprecationWarning, stacklevel=2)
        
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
    """Legacy data module for next activity prediction (deprecated)."""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 x_word_dict: Dict[str, int], y_word_dict: Dict[str, int],
                 max_case_length: int, batch_size: int = 32,
                 num_workers: int = 0):
        import warnings
        warnings.warn("NextActivityDataModule is deprecated. Use CanonicalNextActivityDataModule instead.", 
                     DeprecationWarning, stacklevel=2)
        
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.max_case_length = max_case_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Split train_df into train and validation (legacy approach)
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







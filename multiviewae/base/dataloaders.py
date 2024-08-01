import random
import numpy as np
import pytorch_lightning as pl
import hydra
from torch.utils.data import DataLoader

class MultiviewDataModule(pl.LightningDataModule):
    """LightningDataModule for multi-view data.

    Args:
        n_views (int): Number of views in the data.
        batch_size (int): Batch size.
        is_validate (bool): Whether to use a validation set.
        train_size (float): Proportion of batch to use for training between 0 and 1. Remainder of batch is used for validation.
        dataset (callable): Dataset class to instantiate.
        data (list): Input data. List of torch.Tensors.
        labels (np.array): Dataset labels.
        split_labels (list, optional): List containing 'train' or 'val' for each data point to indicate the split.
    """
    def __init__(
            self,
            n_views,
            batch_size,
            is_validate,
            train_size,
            dataset,
            data,
            labels,
            split_labels=None
        ):
        super().__init__()
        self.n_views = n_views
        self.batch_size = batch_size
        self.is_validate = is_validate
        self.train_size = train_size
        self.data = data
        self.labels = labels
        self.dataset = dataset
        self.split_labels = split_labels
        if not isinstance(self.batch_size, int):
            self.batch_size = self.data[0].shape[0]

    def setup(self, stage):
        if self.is_validate:
            if self.split_labels is not None:
                train_data, val_data, train_labels, val_labels = self.split_data_by_labels()
            else:
                train_data, val_data, train_labels, val_labels = self.train_val_split()
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=train_data, labels=train_labels, n_views=self.n_views)
            self.val_dataset = hydra.utils.instantiate(self.dataset, data=val_data, labels=val_labels, n_views=self.n_views)
        else:
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=self.data, labels=self.labels, n_views=self.n_views)
            self.val_dataset = None
        del self.data

    def train_val_split(self):
        N = self.data[0].shape[0]
        train_idx = list(random.sample(range(N), int(N * self.train_size)))
        val_idx = np.setdiff1d(list(range(N)), train_idx)

        train_data = []
        val_data = []
        for dt in self.data:
            train_data.append(dt[train_idx, :])
            val_data.append(dt[val_idx, :])

        train_labels = None
        val_labels = None
        if self.labels is not None:
            train_labels = self.labels[train_idx]
            val_labels = self.labels[val_idx]

        return train_data, val_data, train_labels, val_labels

    def split_data_by_labels(self):
        train_indices = [i for i, split in enumerate(self.split_labels) if split == 'train']
        val_indices = [i for i, split in enumerate(self.split_labels) if split == 'val']

        train_data = [dt[train_indices, :] for dt in self.data]
        val_data = [dt[val_indices, :] for dt in self.data]

        train_labels = [self.labels[i] for i in train_indices] if self.labels is not None else None
        val_labels = [self.labels[i] for i in val_indices] if self.labels is not None else None

        return train_data, val_data, train_labels, val_labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        if self.is_validate and self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return None

class IndexDataModule(MultiviewDataModule):
    """LightningDataModule for multi-view data.

    Args:
        n_views (int): Number of views in the data.
        batch_size (int): Batch size.
        is_validate (bool): Whether to use a validation set.
        train_size (float): Proportion of batch to use for training between 0 and 1. Remainder of batch is used for validation.
        data (list): Input data. list of identifiers to load data from.
        labels (np.array): Dataset labels. 
    """

    def __init__(
            self,
            n_views, 
            batch_size,
            is_validate,
            train_size,
            dataset,
            data,
            labels,
        ):
        data_ = data[0]
        if not isinstance(batch_size, int):
            batch_size = len(data_)

        super().__init__(n_views=n_views, batch_size=batch_size, is_validate=is_validate, train_size=train_size, 
                         dataset=dataset, data=data_, labels=labels)

    def train_test_split(self):

        N = len(self.data)
        train_idx = list(random.sample(range(N), int(N * self.train_size)))
        test_idx = list(set(list(range(N))) -  set(train_idx))
        data = self.data

        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        train_labels = None
        test_labels = None
        if self.labels is not None:
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]

        return [train_data], [test_data], train_labels, test_labels

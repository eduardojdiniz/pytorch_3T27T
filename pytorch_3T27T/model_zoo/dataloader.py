from torch.utils.data import DataLoader
from .dataset import MNISTDataset
from pytorch_3T27T.base import BaseDataLoader


__all__ = ['MNISTDataLoader']


class MNISTDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, transform, data_dir, batch_size, shuffle,
                 validation_split, nworkers, train=True, worker_init_fn=None,
                 generator=None):

        self.data_dir = data_dir

        self.train_dataset = MNISTDataset(
            self.data_dir,
            train=train,
            download=True,
            transform=transform.get_transform(train=True)
        )
        self.val_dataset = MNISTDataset(
            self.data_dir,
            train=False,
            download=True,
            transform=transform.get_transform(train=False)
        ) if train else None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers,
            'worker_init_fn': worker_init_fn,
            'generator': generator
        }

        super().__init__(self.train_dataset, shuffle=shuffle,
                         **self.init_kwargs)

    def split_validation(self):
        if self.val_dataset is None:
            return None
        else:
            return DataLoader(self.val_dataset, **self.init_kwargs)

#!/usr/bin/env python
# coding=utf-8

"""
This module implements the CycleGAN Dataset class to support the datasets
available at: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
"""

from os.path import join as pjoin
from typing import Any, Callable, Optional, Tuple, TypeVar
from pytorch_3T27T import DATA_ROOT
from .dataset import ImageFolder
from .utils import mkdir, check_integrity

T = TypeVar("T")

__all__ = ['CycleGANDataset']


class CycleGANDataset(ImageFolder):
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets"
    valid_datasets = ["apple2orange", "summer2winter_yosemite", "horse2zebra",
                      "monet2photo", "cezanne2photo", "ukiyoe2photo",
                      "vangogh2photo", "maps", "facades",
                      "iphone2dslr_flower", "ae_photos", "grumpifycat"]

    zip_list = [
        # MD5 Hash                           File name
        ("93831625466b64eaeba58babc52d70f9", "ae_photos.zip"),
        ("5b58c340256288622a835d6f3b6198ae", "apple2orange.zip"),
        ("6516e2926b668312c722dba19bf8ba84", "cezanne2photo.zip"),
        ("87a7be2e97661d730779cf2b791df6eb", "facades.zip"),
        ("cd971b33708c50f02cb2f9f89181883a", "grumpifycat.zip"),
        ("aa76da835f17426278f5cd95c96f957d", "horse2zebra.zip"),
        ("1fc002ebd4cf7083e5ca58f948766582", "iphone2dslr_flower.zip"),
        ("d2e46b2b4b1e4a339ae29cefcc685977", "maps.zip"),
        ("ee5b45684ace8b110a147a12421791c5", "monet2photo.zip"),
        ("a629257ae81b4ab3829d815e97f26e10", "summer2winter_yosemite.zip"),
        ("afceb44d07516786b3024adb775a9050", "ukiyoe2photo.zip"),
        ("2d82c2cd04308c6230f60bb921b769c2", "vangogh2photo.zip"),
    ]

    folder_list = [
        # MD5 Hash                           Folder name
        ("4cef9fb153f89b71c37e6e0eb9e0536f", "ae_photos"),
        ("1fb71ffe5bc80a538767da2ea863f5d8", "apple2orange"),
        ("ae0cb600f13acaad78f1cbb5fbca9531", "cezanne2photo"),
        ("f2c776697aae71724f075e9c14b9aa44", "facades"),
        ("c34ca6a8698dfdd8007574be1dc3b217", "grumpifycat"),
        ("db0e6a498fcfc411d7a7cd26371c2c1d", "horse2zebra"),
        ("92bdb85349dc63a8373fc950756894ee", "iphone2dslr_flower"),
        ("1225f49f749e91bdf2dd9a8341963251", "maps"),
        ("aea31a20541881fa208e33c9302efa91", "monet2photo"),
        ("209cb63e110fc8964b50cc2eeab08b53", "summer2winter_yosemite"),
        ("370ac13b18bce15ec37b19244fd48532", "ukiyoe2photo"),
        ("668fbd6e5f41fe9e691616b1878bf65c", "vangogh2photo"),
    ]

    def __init__(self, root: str = pjoin(DATA_ROOT, "CycleGAN"),
                 dataset: str = "horse2zebra",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 train: bool = True,
                 return_paths: bool = False,
                 max_class_size: int = float('inf'),
                 max_dataset_size: int = float('inf'),
                 download: bool = False) -> None:

        self.root = root
        if dataset == "all":
            self.dataset = self.valid_datasets.copy()
        elif dataset in self.valid_datasets:
            self.dataset = [dataset]
        else:
            raise RuntimeError('Dataset not found.' +
                               f'Valid datasets are {self.valid_datasets}')

        self.zip_dict = {k: v for (v, k) in self.zip_list
                         if k.split('.zip')[0] in self.dataset}

        self.folder_dict = {k: v for (v, k) in self.folder_list
                            if k in self.dataset}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or currupted.' +
                               'You can use download=True to download it')

        if dataset != "all":
            root = pjoin(root, dataset)

        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform,
                         return_paths=return_paths,
                         max_class_size=max_class_size,
                         max_dataset_size=max_dataset_size)

        self.pair_samples()
        self.size_A = len(self.samples['A'])
        self.size_B = len(self.samples['B'])

    def _check_integrity(self) -> bool:
        for (path, md5) in self.folder_dict.items():
            fpath = pjoin(self.root, path)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        import zipfile
        import requests

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        mkdir(self.root)
        for (path, md5) in self.zip_dict.items():
            fpath_zip = pjoin(self.root, path)
            r = requests.get(pjoin(self.url, path), allow_redirects=True)
            with open(fpath_zip, 'wb') as f:
                f.write(r.content)
            if not check_integrity(fpath_zip, md5):
                return False

            with zipfile.ZipFile(fpath_zip, "r") as f:
                f.extractall(self.root)

    def pair_samples(self):
        if self.train:
            class_to_domain = {'trainA': 'A', 'trainB': 'B'}
        else:
            class_to_domain = {'testA': 'A', 'testB': 'B'}

        idx_to_class = {v: k for k, v in self.class_to_idx.items()
                        if k in class_to_domain.keys()}
        idx_to_domain = {k: class_to_domain[v]
                         for k, v in idx_to_class.items()}

        samples = {}
        for i, domain in enumerate(['A', 'B']):
            samples[domain] = [(s[0], i) for s in self.samples
                               if s[1] in idx_to_domain.keys() and
                               idx_to_domain[s[1]] == domain]
        self.samples = samples

        return self

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        Parameters
        ----------
        idx : int
            A (random) integer for data intexing

        Returns
        -------
        _: Tuple[Any, ...]
            (sample, target) where target is class_index of the target class.
            (sample, target, path) if ``self.return_paths`` is True
        """

        # Make sure indexes are within A and B ranges
        path_A, target_A = self.samples['A'][idx % self.size_A]
        path_B, target_B = self.samples['B'][idx % self.size_B]
        sample_A, sample_B = self.loader(path_A), self.loader(path_B)
        if self.transform is not None:
            sample_A = self.transform(sample_A)
            sample_B = self.transform(sample_B)
        if self.target_transform is not None:
            target_A = self.target_transform(target_A)
            target_B = self.target_transform(target_B)
        if not self.return_paths:
            return {'A': (sample_A, target_A, path_A),
                    'B': (sample_B, target_B, path_B)}

        return {'A': (sample_A, target_A), 'B': (sample_B, target_B)}

    def __len__(self) -> int:
        """
        As we have two datasets with potentially different number of images, we
        take a maximum of
        """

        return max(self.size_A, self.size_B)


class Cityscapes(ImageFolder):
    url = ["https://www.cityscapes-dataset.com/file-handling/?packageID=1",
           "https://www.cityscapes-dataset.com/file-handling/?packageID=3"]

    zip_list = [
        # MD5 hash                           # File name
        ("4237c19de34c8a376e9ba46b495d6f66", "gtFine_trainvaltest.zip"),
        ("0a6e97e94b616a514066c9e2adb0c97f", "leftImg8bit_trainvaltest.zip"),
    ]

    folder_list = [
        # MD5 hash                           # Folder name
        ("4237c19de34c8a376e9ba46b495d6f66", "gtFine_trainvaltest"),
        ("0a6e97e94b616a514066c9e2adb0c97f", "leftImg8bit_trainvaltest"),
    ]

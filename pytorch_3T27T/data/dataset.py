#!/usr/bin/env python
# coding=utf-8

"""
Datasets are based on the PyTorch ``torch.utils.data.Dataset`` data
primitive. They store the samples and their corresponding labels. Pytorch
domain libraries (e.g., vision, text, audio) provide pre-loaded datasets (e.g.,
MNIST) that subclass ``torch.utils.data.Dataset`` and implement functions
specific to the particular data. They can be used to prototype and benchmark
your model. You can find them at
[Image Datasets](https://pytorch.org/vision/stable/datasets.html),
[Text Datasets](https://pytorch.org/text/stable/datasets.html), and
[Audio Datasets](https://pytorch.org/audio/stable/datasets.html).

This module implements an abstract base class `BaseVisionDataset` for vision
datasets. It also replicates the official PyTorch image folder
(https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so it can inherent from `BaseVisionDataset` and have extended functionality.
"""

import os
from os.path import join as pjoin
from itertools import zip_longest
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar
from torchvision.datasets.vision import VisionDataset
from pytorch_3T27T import DATA_ROOT
from .utils import IMG_EXTENSIONS, is_valid_extension, default_image_loader

T_co = TypeVar('T_co', covariant=True)
T = TypeVar("T")

__all__ = ['BaseVisionDataset', 'FolderDataset', 'ImageFolder']


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    See :class:`FolderDataset` for details.
    """

    classes = sorted([entry.name for entry in os.scandir(directory)
                      if entry.is_dir()])

    if not classes:
        msg = f"Couldn't find any class folder in {directory}."
        raise FileNotFoundError(msg)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(directory: str,
                 class_to_idx: Optional[Dict[str, int]] = None,
                 extensions: Optional[Tuple[str, ...]] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 max_class_size: int = float("inf"),
                 max_dataset_size: int = float("inf"),
                 ) -> List[Tuple[str, int]]:
    """
    Generates a list of samples of a form (path_to_sample, class).

    See :class:`FolderDataset` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of
    the ``find_classes`` function by default.
    """

    # Arguments parsing
    directory = os.path.expanduser(directory)
    assert os.path.isdir(directory) or os.path.islink(directory), \
        f'{directory} is not a valid directory'

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_idx' must have at least one entry.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        msg = ("Both 'extensions' and 'is_valid_file' cannot be None ",
               "or not None at the same time")
        raise ValueError(msg)

    if extensions is not None:
        def is_valid_file(fname: str) -> bool:
            return is_valid_extension(fname, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    # Main logic
    instances_dict = {}
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[target_class]
        class_instances = []
        n_instances = 0
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = pjoin(root, fname)
                    item = path, class_idx
                    class_instances.append(item)
                    n_instances += 1
                    available_classes.add(target_class)

        # Make sure to respect the maximum number of samples per class
        instances_dict[target_class] = class_instances[:min(max_class_size,
                                                       n_instances)]

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = ("Found no valid file for the classes ",
               f"{', '.join(sorted(empty_classes))}. ",
               f"Supported extensions are: {', '.join(IMG_EXTENSIONS)}")
        raise FileNotFoundError(msg)

    instances = []
    # Make sure to respect the maximum number of samples allowed in the dataset
    for samples in zip_longest(*sorted(instances_dict.values())):
        if len(instances) > len(available_classes) * (max_dataset_size - 1):
            break
        # Remove None values using filter
        instances.extend(list(filter(None, samples)))

    return instances


class BaseVisionDataset(VisionDataset, metaclass=ABCMeta):
    """
    Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    To create a subclass, you need to implement the following functions:

    <__init__>:
        (Optionally) Initialize the class, first call super.__init__(root,
        train, transform, target_transform, **kwargs).
    <__len__>:
        Return the number of samples in the dataset.
    <__getitem__>:
        Get a data point.
    """

    def __init__(self, root: str = DATA_ROOT, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 max_class_size: int = float('inf'),
                 max_dataset_size: int = float('inf'),
                 **kwargs) -> None:
        """
        Arguments
        ---------
        root : str
            Root directory of dataset.
        train : bool
            If True, creates dataset from training samples, otherwise, from
            test samples.
        transform : Optional[Callable]
            A function/transform that takes in an PIL image and returns a
            transformed version, e.g, ``torchvision.transforms.RandomCrop``
        target_transform : Optional[Callable]
            A function/transform that takes in the target and transforms it.

        """

        super().__init__(root=root, transform=transform,
                         target_transform=target_transform)
        self.train = train
        self.max_class_size = max_class_size
        self.max_dataset_size = max_dataset_size

    @abstractmethod
    def default_params(self):
        """Set the default train and test arguments and options"""

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Arguments
        ---------
        idx : int
            Index

        Returns
        -------
        _ : Any
            Sample and meta data, optionally transformed by the respective
            transforms.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns
        -------
        _ : int
            Number of samples in dataset
        """


class FolderDataset(BaseVisionDataset):
    """
    A generic folder dataset.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` and :meth:`make_dataset` methods.

    Attributes
    ----------
    classes : list
        List of the class names sorted alphabetically.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    samples : list
        List of (sample, class_index) tuples
    targets : list
        The class_index value for each image in the dataset
    """

    def __init__(self, root: str, loader: Callable[[str], Any],
                 train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extensions: Optional[Tuple[str, ...]] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 return_paths: bool = False,
                 max_class_size: int = float('inf'),
                 max_dataset_size: int = float('inf'),
                 **kwargs) -> None:
        """
        Initialize the class

        Parameters
        ----------
        root : str
            Data root directory
        loader : Callable
            A function to load a sample given its path.
        train : bool
            Whether in training mode or test mode. You can use this flag to add
            training-specific or test-specific configurations. Default: True.
        transform : Optional[Callable]
            A function/transform that takes in a sample and returns a
            transformed version, e.g, ``torchvision.transforms.RandomCrop`` for
            images.
        target_transform : Callable[Optional]
            Optional function/transform that takes in a target and returns a
            transformed version.
        extensions : Tuple[str]
            A list of allowed extensions.
        is_valid_file :  Optional[Callable[[str], bool]]
            A function that takes path of a file and check if the file is a
            valid file (used to check of corrupt files).
        return_paths : bool
            If True, calling the dataset returns `(sample, target), target,
            sample path` instead of returning `(sample, target), target`

        Notes
        -----
        Both `extensions` and `is_valid_file` cannot be None or not None at the
        same time.
        """

        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform,
                         max_class_size=max_class_size,
                         max_dataset_size=max_dataset_size, **kwargs)

        self.train = train
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions,
                                    is_valid_file, self.max_class_size,
                                    self.max_dataset_size)

        if len(samples) == 0:
            msg = (f'Found 0 samples in: {root}. \n Supported ',
                   f'extensions are: {",".join(IMG_EXTENSIONS)}')
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.return_paths = return_paths

    @staticmethod
    def make_dataset(directory: str,
                     class_to_idx: Dict[str, int],
                     extensions: Optional[Tuple[str, ...]] = None,
                     is_valid_file: Optional[Callable[[str], bool]] = None,
                     max_class_size: int = float("inf"),
                     max_dataset_size: int = float("inf"),
                     ) -> List[Tuple[str, int]]:
        """
        Generates a list of images of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file
        instead of from the disk.

        Parameters
        ----------
        directory : str
            root dataset directory, corresponding to ``self.root``.
        class_to_idx : Dict[str, int]
            Dictionary mapping class name to class index.
        extensions : Tuple[str]
            A list of allowed extensions.
        is_valid_file :  Optional[Callable[[str], bool]]
            A function that takes path of a file and check if the file is a
            valid file (used to check of corrupt files).
        max_dataset_size : int
            Maximum number of samples allowed in the dataset
        max_class_size : int
            Maximum number of samples allowed per class

        Raises
        ------
        ValueError: In case ``class_to_idx`` is empty.
        FileNotFoundError: In case no valid file was found for any class.

        Returns
        -------
        _: List[Tuple[str, int]]
            Samples of a form (path_to_sample, class)

        Notes
        -----
        Both `extensions` and `is_valid_file` cannot be None or not None at the
        same time.
        """

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the
            # class_to_idx logic of the find_classes() function, instead of
            # using that of the find_classes() method, which is potentially
            # overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions,
                            is_valid_file=is_valid_file,
                            max_class_size=max_class_size,
                            max_dataset_size=max_dataset_size)

    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a image dataset structured as follows:

        directory/
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── ...
        │   └── xxz.ext
        └── class_y
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext


        This method can be overridden to only consider a subset of classes,
        or to adapt to a different dataset directory structure.

        Arguments
        ---------
        directory : str
            Root directory path, corresponding to ``self.root``

        Raises
        ------
        FileNotFoundError: If ``directory`` has no class folders.

        Returns
        -------
        _: Tuple[List[str], Dict[str, int]]
            List of all classes and dictionary mapping each class to an index.
        """

        return find_classes(directory)

    def default_params(self):
        default_train_args = []
        default_train_options = {
            'mode': "unaligned",
            'max_dataset_size': float("inf"),
            'max_class_size': float("inf"),
        }

        default_test_args = []
        default_test_options = {
            'mode': "aligned",
            'max_dataset_size': float("inf"),
            'max_class_size': float("inf"),
        }

        if self.train:
            return default_train_args, default_train_options
        return default_test_args, default_test_options

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

        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return sample, target, path
        return sample, target

    def __len__(self) -> int:
        """Return the total number of images"""

        return len(self.samples)


class ImageFolder(FolderDataset):
    """
    A generic image folder dataset where the images are arranged in this way by
    default:

    root/
    ├── dog
    │   ├── xxx.png
    │   ├── xxy.png
    │   └── ...
    │   └── xxz.png
    └── cat
        ├── 123.png
        ├── nsdf3.png
        └── ...
        └── asd932_.png

    This class inherits from :class:`FolderDataset` so the same methods can be
    overridden to customize the dataset.

    Attributes
    ----------
    classes : list
        List of the class names sorted alphabetically.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    samples : list
        List of (images, class_index) tuples
    targets : list
        The class_index value for each image in the dataset
    """

    def __init__(self, root: str,
                 loader: Optional[Callable[[str], Any]] = default_image_loader,
                 train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 return_paths: bool = False,
                 max_class_size: int = float('inf'),
                 max_dataset_size: int = float('inf'),
                 **kwargs) -> None:
        """
        Initialize the class

        Parameters
        ----------
        root : str
            Data root directory
        loader : Optional[Callable]
            A function to load a image given its path.
        train : bool
            Whether in training mode or test mode. You can use this flag to add
            training-specific or test-specific configurations. Default: True.
        transform : Optional[Callable]
            A function/transform that takes in an PIL image and returns a
            transformed version, e.g, ``torchvision.transforms.RandomCrop``
        target_transform : Callable[Optional]
            Optional function/transform that takes in a target and returns a
            transformed version.
        is_valid_file : Optional[Callable[[str], bool]]
            A function that takes path of an image file and check if the file
            is a valid image file (used to check of corrupt files).
        return_paths : bool
            If True, calling the dataset returns `(img, label), label, image
            path` instead of returning `(img, label), label`
        """
        super().__init__(root=root, loader=loader, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         extensions=IMG_EXTENSIONS if is_valid_file is None
                         else None, is_valid_file=is_valid_file,
                         return_paths=return_paths,
                         max_class_size=max_class_size,
                         max_dataset_size=max_dataset_size, **kwargs)

    def default_params(self):
        default_train_args = []
        default_train_options = {
            'max_dataset_size': float("inf"),
            'max_class_size': float("inf"),
        }

        default_test_args = []
        default_test_options = {
            'max_dataset_size': float("inf"),
            'max_class_size': float("inf"),
        }

        if self.train:
            return default_train_args, default_train_options
        return default_test_args, default_test_options

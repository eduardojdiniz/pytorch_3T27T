#!/usr/bin/env python
# coding=utf-8

"""
Data does not always come in its final processed form that is required for
training machine learning algorithms. We use transforms to perform some
manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters - `transform` to modify the
features and `target_transform` to modify the labels - that accept callables
containing the transformation logic. The `torchvision.transforms` module offers
several commonly-used transforms out of the box.

Transforms can be chained together using `torchvision.Transforms.Compose`. The
transformations that accept tensor images also accept batches of tensor images.
A Tensor Image is a tensor with (C, H, W) shape, where C is a number of
channels, H and W are image height and width. A batch of Tensor Images is a
tensor of (B, C, H, W) shape, where B is a number of images in the batch.

The expected range of the values of a tensor image is implicitely defined by
the tensor dtype. Tensor images with a float dtype are expected to have values
in [0, 1). Tensor images with an integer dtype are expected to have values in
[0, MAX_DTYPE] where MAX_DTYPE is the largest value that can be represented in
that dtype.
"""

from abc import ABC, abstractmethod
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple


__all__ = ['AugmentationFactory', 'BaseTransform', 'print_size_warning']


AUGMENTATIONS = [
    'grayscale', 'fixsize', 'resize', 'scale_width', 'scale_shortside', 'zoom',
    'crop', 'patch', 'trim', 'flip', 'convert', 'make_power_2'
]

# bicubic = InterpolationMode.BICUBIC
bicubic = Image.BICUBIC


class AugmentationFactory:
    """Augmentation factory"""

    def __init__(self, train: bool = True) -> None:
        self.train = train

    def default_params(self):
        default_train_args = []
        default_train_options = {
            'out_ch': 1,
            'load_size': (256, 256),
            'out_size': (128, 128),
            'max_size': (128, 128),
            'crop_size': (128, 128),
            'crop_pos': None,
            'trim_size': (128, 128),
            'patch_size': (128, 128),
            'patch_stride': 0,
            'shortside_size': 128,
            'zoom_factor': None,
            'flip': True,
            'power_base': 4,
            'interp': bicubic,
            'dataset_name': 'horse2zebra'
        }

        default_test_args = []
        default_test_options = {
            'out_ch': 1,
            'load_size': (256, 256),
            'out_size': (128, 128),
            'max_size': (128, 128),
            'crop_size': (128, 128),
            'crop_pos': (0, 0),
            'trim_size': (128, 128),
            'patch_size': (128, 128),
            'patch_stride': 0,
            'shortside_size': 128,
            'zoom_factor': (1, 1),
            'power_base': 4,
            'interp': bicubic,
            'dataset_name': 'horse2zebra'
        }

        if self.train:
            return default_train_args, default_train_options
        return default_test_args, default_test_options

    def get_transform(self):
        augmentations, options = self.cfg.args, self.cfg.options
        transform = {}

        interp = options['interp']
        if 'grayscale' in augmentations:
            transform['grayscale'] = T.Grayscale(options['out_ch'])

        if 'fixsize' in augmentations:
            out_size = options['out_size']
            transform['fixsize'] = T.Resize(out_size, interpolation=interp)

        if 'resize' in augmentations:
            load_size = options['load_size']
            if options['dataset_name'] == "gta2cityscapes":
                load_size[0] = load_size[0] // 2
            transform['resize'] = T.Resize(load_size, interpolation=interp)

        elif 'scale_width' in augmentations:
            width, _ = options['out_size']
            _, max_height = options['max_size']

            def lambd(img): return self.scale_width(img, width, max_height,
                                                    interp=interp)
            transform['scale_width'] = T.Lambda(lambd)

        elif 'scale_shortside' in augmentations:
            size = options['shortside_size']

            def lambd(img):
                return self.scale_shortside(img, size, interp=interp)
            transform['scale_shortside'] = T.Lambda(lambd)

        if 'zoom' in augmentations:
            max_size = options['max_size']
            factor = options['zoom_factor']

            def lambd(img): return self.random_zoom(img, max_size,
                                                    factor=factor,
                                                    interp=interp)
            transform['zoom'] = T.Lambda(lambd)

        if 'crop' in augmentations:
            size = options['crop_size']
            pos = options['crop_pos']
            if pos is None:
                transform['crop'] = T.RandomCrop(size=size)
            else:
                def lambd(img): return self.crop(img, size=size, pos=pos)
                transform['crop'] = T.Lambda(lambd)

        if 'patch' in augmentations:
            size = options['patch_size']
            stride = options['patch_stride']
            def lambd(img): return self.patch(img, size, stride=stride)
            transform['patch'] = T.Lambda(lambd)

        if 'trim' in augmentations:
            def lambd(img): return self.trim(img, options['trim_size'])
            transform['trim'] = T.Lambda(lambd)

        if 'flip' in augmentations:
            if options['flip'] is None:
                transform['flip'] = T.RandomHorizontalFlip()
            else:
                def lambd(img): return self.flip(img, flip=options['flip'])
                transform['flip'] = T.Lambda(lambd)

        if 'convert' in augmentations:
            grayscale = True if 'grayscale' in augmentations else False
            def lambd(img): return self.convert(img, grayscale=grayscale)
            transform['convert'] = T.Lambda(lambd)

        # Make power of 'base'
        base = options['power_base']
        def lambd(img): return self.make_power_2(img, base=base, interp=interp)
        transform['make_power_2'] = T.Lambda(lambd)

        transform_list = []
        for aug in augmentations:
            if isinstance(transform[aug], list):
                transform_list.extend(transform[aug])
            else:
                transform_list.append(transform[aug])

        transform_list.append(T.ToTensor())
        return T.Compose(transform_list)

    @staticmethod
    def scale_width(img, width, max_height, interp=bicubic):
        augment = ScaleWidthTransform(width, max_height, interp=interp)
        return augment(img)

    @staticmethod
    def scale_shortside(img, size, interp=bicubic):
        augment = ScaleShortsideTransform(size, interp=interp)
        return augment(img)

    @staticmethod
    def random_zoom(img, max_size, factor=None, interp=bicubic):
        augment = RandomZoomTransform(max_size, factor=factor, interp=interp)
        return augment(img)

    @staticmethod
    def crop(img, size, pos):
        augment = CropTransform(size=size, pos=pos)
        return augment(img)

    @staticmethod
    def patch(img, size, stride: int = 0):
        augment = PatchTransform(size, stride=stride)
        return augment(img)

    @staticmethod
    def trim(img, size):
        augment = TrimTransform(size)
        return augment(img)

    @staticmethod
    def make_power_2(img, base: int = 4, interp=bicubic):
        augment = MakePowerTwoTransform(base=base, interp=interp)
        return augment(img)

    @staticmethod
    def flip(img, flip):
        augment = FlipTransform(flip=flip)
        return augment(img)

    @staticmethod
    def convert(img, grayscale: bool = False):
        augment = ConvertTransform(grayscale=grayscale)
        return augment(img)


class BaseTransform(ABC):
    """ Base class for all custom transforms"""

    def __init__(self):
        """Init"""

    @abstractmethod
    def __call__(self, sample):
        """Call"""

    def __repr(self):
        return self.__class__.__name__ + '()'


class IdentityTransform(BaseTransform):
    """ Return Image unchanged"""

    def __call__(self, img):
        return img


class RandomZoomTransform(BaseTransform):
    """ Random Zoom Transform"""

    def __init__(self, max_size: Tuple[int, int], factor: bool = None,
                 interp=bicubic):
        self.max_size = max_size
        self.factor = factor
        self.interp = interp

    def __call__(self, img):
        """Call"""

        if self.factor is None:
            zoom_w, zoom_h = np.random.uniform(0.8, 1.0, size=[2])
        else:
            zoom_w, zoom_h = self.factor
        in_w, in_h = img.size
        max_w, max_h = self.max_size
        zoom_w = max(max_w, in_w * zoom_w)
        zoom_h = max(max_h, in_h * zoom_h)
        img = img.resize((int(round(zoom_w)), int(round(zoom_h))), self.interp)
        return img


class ConvertTransform(BaseTransform):
    """ Convert to Tensor and Normalize Transform"""

    def __init__(self, grayscale: bool = False):
        if grayscale:
            self.mean, self.std = (0.5,), (0.5,)
        else:
            self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    def __call__(self, img):
        # Enforce single channel if img is grayscale
        if hasattr(img, 'mode'):
            if img.mode == "L":
                self.mean, self.std = (0.5,), (0.5,)
        elif hasattr(img, 'shape'):
            if len(img.shape) == 2:
                self.mean, self.std = (0.5,), (0.5,)

        # T.Normalize does not support PIL Image, convert to Torch Tensor first
        transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

        return transform(img)


class ScaleWidthTransform(BaseTransform):
    """ Scale Width Transform"""

    def __init__(self, width: int, max_height: int, interp=bicubic):
        self.width = width
        self.max_height = max_height
        self.interp = interp

    def __call__(self, img):
        in_w, in_h = img.size
        if in_w == self.width and in_h >= self.max_height:
            return img
        w = self.width
        h = int(max(self.width * in_h / in_w, self.max_height))
        return img.resize((w, h), self.interp)


class ScaleShortsideTransform(BaseTransform):
    """ Scale Shortside Transform"""

    def __init__(self, size, interp=bicubic):

        self.size = size
        self.interp = interp

    def __call__(self, img):
        in_w, in_h = img.size
        shortside = min(in_w, in_h)
        if shortside >= self.size:
            return img
        scale = self.size / shortside
        w = round(in_w * scale)
        h = round(in_h * scale)
        return img.resize((w, h), self.interp)


class CropTransform(BaseTransform):
    """ Crop Transform"""

    def __init__(self, size: Tuple[int, int] = None,
                 pos: Tuple[int, int] = None):
        self.size = size
        self.pos = pos

    def __call__(self, img):
        in_w, in_h = img.size
        pos_x, pos_y = self.pos
        out_w, out_h = self.size
        if (in_w > out_w or in_h > out_h):
            return img.crop((pos_x, pos_y, pos_x + out_w, pos_y + out_h))
        return img


class PatchTransform(BaseTransform):
    """ Patch Transform"""

    def __init__(self, size: tuple, stride: int = 0):
        self.size = size
        self.stride = stride

    def __call__(self, img):
        in_w, in_h = img.size
        patch_w, patch_h = self.size
        num_w, num_h = in_w // patch_w,  in_h // patch_h
        room_x = in_w - num_w * patch_w
        room_y = in_h - num_h * patch_h
        x_start = np.random.randint(int(room_x) + 1)
        y_start = np.random.randint(int(room_y) + 1)
        idx = self.stride % (num_w * num_h)
        idx_x, idx_y = idx // num_h, idx % num_w
        pos_x = x_start + idx_x * patch_w
        pos_y = y_start + idx_y * patch_h
        return img.crop((pos_x, pos_y, pos_x + patch_w, pos_y + patch_h))


class FlipTransform(BaseTransform):
    """ Flip Transform"""

    def __init__(self, flip: bool = True):
        self.flip = flip

    def __call__(self, img):
        if self.flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class TrimTransform(BaseTransform):
    """ Trim Transform"""

    def __init__(self, size: int = 256):
        self.size = size

    def __call__(self, img):
        in_w, in_h = img.size
        trim_w, trim_h = self.size
        if in_w > trim_w:
            start_x = np.random.randint(in_w - trim_w)
            end_x = start_x + trim_w
        else:
            start_x = 0
            end_x = in_w
        if in_h > trim_h:
            start_y = np.random.randint(in_h - trim_h)
            end_y = start_y + trim_h
        else:
            start_y = 0
            end_y = in_h
        return img.crop((start_x, start_y, end_x, end_y))


class MakePowerTwoTransform(BaseTransform):
    """ Make Power 2 Transform"""

    def __init__(self, base: int = 4, interp=bicubic):
        self.base = base
        self.interp = interp

    def __call__(self, img):
        in_w, in_h = img.size
        h = int(round(in_h / self.base) * self.base)
        w = int(round(in_w / self.base) * self.base)
        if h == in_h and w == in_w:
            return img
        print_size_warning(in_w, in_h, w, h)
        return img.resize((w, h), self.interp)


def print_size_warning(in_w, in_h, w, h):
    """Print warning information about image size (only print once)"""

    if not hasattr(print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (in_w, in_h, w, h))
        print_size_warning.has_printed = True

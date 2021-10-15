#!/usr/bin/env python
# coding=utf-8

"""This module contains simple helper functions """


from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torchvision.transforms as T


__all__ = [
    'str2bool', 'torch2numpy', 'correct_resize', 'correct_resize_label',
    'mkdirs', 'mkdir', 'print_numpy', 'save_image', 'diagnose_network'
]


def str2bool(string):
    """"
    Converts a str into bool

    Parameters
    ----------
    string : str

    Returns
    -------
    boolean : bool
    """

    if isinstance(string, bool):
        boolean = string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        boolean = True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        boolean = False
    else:
        raise TypeError('TypeError: Boolean value expected.')

    return boolean


def torch2numpy(input_image, imtype=np.uint8):
    """"
    Converts a torch.Tensor array into a numpy.ndarray image array

    Parameters
    ----------
    input_image : torch.Tensor
    imtype : numpy.dtype
        The desired type of the converted numpy array

    Parameters
    ----------
    image_numpy : numpy.ndarray
    """

    if not isinstance(input_image, np.ndarray):
        # Get the data from a variable
        if isinstance(input_image, torch.Tensor):
            image_torch = input_image.data
        else:
            return input_image
        # Convert it into a numpy.ndarray
        image_numpy = image_torch[0].cpu().float().numpy()
        # grayscale to RGB
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # Post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # if it is a numpy array, do nothing
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def correct_resize_label(tensor, size):
    """
    Resize torch.Tensor label

    Parameters
    ----------
    tensor : torch.Tensor
    size : Tuple[int]

    Returns
    -------
    resized_tensor : torch.Tensor
    """

    device = tensor.device
    tensor = tensor.detach().cpu()
    resized = []
    for i in range(tensor.size(0)):
        one_tensor = tensor[i, :1]
        one_numpy = np.transpose(one_tensor.numpy().astype(np.uint8),
                                 (1, 2, 0))
        one_numpy = one_numpy[:, :, 0]
        one_image = Image.fromarray(one_numpy).resize(size, Image.NEAREST)
        resized_one_tensor = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_one_tensor)
    resized_tensor = torch.stack(resized, dim=0).to(device)
    return resized_tensor


def correct_resize(tensor, size, mode=Image.BICUBIC):
    """
    Resize torch.Tensor image

    Parameters
    ----------
    tensor : torch.Tensor
    size : Tuple[int]
    mode : PIL.Image Filter
        Resampling filter. Default: PIL.Image.BICUBIC

    Returns
    -------
    resized_tensor : torch.Tensor
    """

    device = tensor.device
    tensor = tensor.detach().cpu()
    resized = []
    for i in range(tensor.size(0)):
        one_tensor = tensor[i:i + 1]
        one_image = Image.fromarray(torch2numpy(one_tensor))
        one_image = one_image.resize(size, Image.BICUBIC)
        resized_one_tensor = T.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_one_tensor)
    resized_tensor = torch.stack(resized, dim=0).to(device)
    return resized_tensor


def diagnose_network(net, name='network'):
    """
    Calculate and print the mean of average absolute gradients

    Parameters
    ----------
    net : torch.device
        Torch network
    name : str
        The name of the network
    """

    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Save a numpy.ndarray image to the disk

    Parameters
    ----------
    image_numpy : numpy.ndarray
        Input numpy array
    image_path : str
        The path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """
    Print the mean, min, max, median, std, and size of a numpy array

    Parameters
    ----------
    val : bool
        If print the values of the numpy array
    shp : bool
        If print the shape of the numpy array
    """

    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        msg = f'mean = {np.mean(x):3.3f}, min = {np.min(x):3.3f}, '
        msg += f'max = {np.max(x):3.3f}, median = {np.median(x):3.3f}, '
        msg += f'std = {np.std(x):3.3f}'
        print(msg)


def mkdirs(paths):
    """
    Create empty directories if they don't exist

    Parameters
    ----------
    paths : List[str]
        A list of directory paths
    """

    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Create a single empty directory if it didn't exist

    Parameters
    ----------
    path : str
        A single directory path
    """

    if not os.path.exists(path):
        os.makedirs(path)

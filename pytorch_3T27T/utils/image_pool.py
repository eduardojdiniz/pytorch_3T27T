#!/usr/bin/env python
# coding=utf-8

import random
import torch

__all__ = ['ImagePool']


class ImagePool:
    """
    This class implements an image buffer that stores previously generated
    images. This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """
        Initialize the ImagePool class

        Parameters
        ----------
        pool_size : int
            The size of image buffer, if pool_size=0, no buffer will be created
        """

        self.pool_size = pool_size

        # if pool_size bigger than zero, create an empty pool
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Return an image from the pool.

        Parameters
        ----------
        images : torch.Tensor
            The latest generated images from the generator

        Returns
        -------
        return_images : torch.Tensor
            Images from the buffer. If buffer is not full, will return the
            given input image. Else, wth 50% chance, a previously stored image
            in the buffer will be returned and the given input image stored in
            the buffer.
        """

        # If the buffer size is 0, do nothing
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # If the buffer is not full keep inserting images to the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                # With 50% chance, the buffer will return a previously stored
                # image, and insert the current image into the buffer
                if p > 0.5:
                    # randint is inclusive
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                # Else, the buffer will return the current image
                else:
                    return_images.append(image)

        # Collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images

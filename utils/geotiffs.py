import numbers
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image

from . import patch_iter


class Geotiffs(Dataset):
    def __init__(self, root, patch_shape, steps=None, transform=None, **kwargs):
        """

        Parameters
        ==========

        steps: tuple of ints
            step width to move to the next patch. Can be used to get some overlap between patches.
            If steps is None patch_shape is uses as steps, meaning zero overlap.

        kwargs: labels and their corresponding tif-files.
            __getitem__ returns a dictionary with the same labels as keys and
            patches for the respective tif-files.

        """
        self.root = pathlib.Path(root)

        # tif files that make up the data set
        self.tifs = kwargs

        # check that all tif files have the same dimension, i.e. height and width
        self.shape = self._check_dims()

        if isinstance(patch_shape, numbers.Number):
            patch_shape = (int(patch_shape), int(patch_shape))

        if isinstance(steps, numbers.Number):
            steps = (steps, ) * len(patch_shape)
        elif steps is None:
            steps = patch_shape

        if not len(steps) == len(patch_shape):
            raise ValueError("`steps` is incompatible with `patch_shape`")

        self.steps = steps
        self.patch_shape = patch_shape

        self.indices = list(patch_iter.patch_index_tuples(self.patch_shape, self.shape, steps=self.steps))
        
        print(len(self.indices))

        # managed by __enter__ and __exit__
        self.opened_tifs = None

        self.transform = transform

    def _open_tifs(self):
        opened_tifs = {
            label: rasterio.open(str(self.root / tif_name), 'r')
            for label, tif_name in self.tifs.items()
        }
        return opened_tifs

    @staticmethod
    def _close_tifs(opened_tifs):
        for tif_file in opened_tifs.values():
            tif_file.close()

    def _check_dims(self):
        opened_tifs = self._open_tifs()

        shapes = set(dst.shape for dst in opened_tifs.values())
        if len(shapes) != 1:
            raise ValueError('shapes for tif files do not match')

        self._close_tifs(opened_tifs)

        # return only element, i.e. the spatial dimensions
        return shapes.pop()

    def __enter__(self):
        self.opened_tifs = self._open_tifs()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._close_tifs(self.opened_tifs)
        self.opened_tifs = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index_tuple = self.indices[index]
        window = rasterio.windows.Window.from_slices(*index_tuple)

        patch = {}
        for label, tif_file in self.opened_tifs.items():
            data = tif_file.read(window=window)[0]
            data = Image.fromarray(data)
            patch[label] = data

        if self.transform:
            patch = self.transform(patch)

        return patch

import os

import torch

from torch.utils import data
from torchvision.datasets.folder import (default_loader,
                                         has_file_allowed_extension,
                                         IMG_EXTENSIONS)


class UnlabelledImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = self.find_images(os.path.join(root))

    def __getitem__(self, index):
        path = self.paths[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        # Add a bogus label to be compatible with standard image datasets.
        return image, torch.tensor([0.])

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def find_images(dir):
        paths = []
        for fname in sorted(os.listdir(dir)):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(dir, fname)
                paths.append(path)
        return paths


def load_num_batches(loader, num_batches):
    """A generator that returns num_batches batches from the loader, irrespective of the length
    of the dataset."""
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return

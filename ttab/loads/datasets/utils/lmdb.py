# -*- coding: utf-8 -*-
import os
import sys

import cv2
import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image

from .serialize import loads

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def be_ncwh_pt(x):
    return x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)


def uint8_to_float(x):
    x = x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)
    return x.float() / 128.0 - 1.0


class LMDBPT(data.Dataset):
    """A class to load the LMDB file for extreme large datasets.
    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.lmdb_files = self._get_valid_lmdb_files()

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(
                LMDBPTClass(
                    root=lmdb_file,
                    transform=transform,
                    target_transform=target_transform,
                    is_image=is_image,
                )
            )

        # build up indices.
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]
        self._build_indices()
        self._prepare_target()

    def _get_valid_lmdb_files(self):
        """get valid lmdb based on given root."""
        if not self.root.endswith(".lmdb"):
            files = []
            for l in os.listdir(self.root):
                if "_" in l and "-lock" not in l:
                    files.append(os.path.join(self.root, l))
            return files
        else:
            return [self.root]

    def _build_indices(self):
        self.from_to_indices = enumerate(zip(self.indices[:-1], self.indices[1:]))

    def _get_matched_index(self, index):
        if len(list(self.from_to_indices)) == 0:
            return 0, index

        for ind, (from_index, to_index) in self.from_to_indices:
            if from_index <= index and index < to_index:
                return ind, index - from_index

    def __getitem__(self, index, apply_transform=True):
        block_index, item_index = self._get_matched_index(index)
        image, target = self.dbs[block_index].__getitem__(item_index, apply_transform)
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _prepare_target(self):
        cache_file = self.root + "_targets_cache_"
        if os.path.isfile(cache_file):
            self.targets = pickle.load(open(cache_file, "rb"))
        else:
            self.targets = [
                self.__getitem__(idx, apply_transform=False)[1]
                for idx in range(self.length)
            ]
            pickle.dump(self.targets, open(cache_file, "wb"))


class LMDBPTClass(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_image = is_image

        # init the placeholder for env and length.
        self.env = None
        self.length = self._get_tmp_length()

    def _open_lmdb(self):
        return lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True,
            lock=False,
            readahead=False,
            # map_size=1099511627776 * 2,
            max_readers=1,
            meminit=False,
        )

    def _get_tmp_length(self):
        env = lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True,
            lock=False,
            readahead=False,
            # map_size=1099511627776 * 2,
            max_readers=1,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            length = txn.stat()["entries"]

            if txn.get(b"__keys__") is not None:
                length -= 1
        # clean everything.
        del env
        return length

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

            if txn.get(b"__keys__") is not None:
                self.length -= 1

    def _prepare_cache(self):
        cache_file = self.root + "_cache_"
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor() if key != b"__keys__"]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _decode_from_image(self, x):
        image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype("uint8")
        return Image.fromarray(image, "RGB")

    def _decode_from_array(self, x):
        return Image.fromarray(x.reshape(3, 32, 32).transpose((1, 2, 0)), "RGB")

    def __getitem__(self, index, apply_transform=True):
        if self.env is None:
            # # open lmdb env.
            self.env = self._open_lmdb()
            # # get file stats.
            # self._get_length()
            # # prepare cache_file
            self._prepare_cache()

        # setup.
        env = self.env
        with env.begin(write=False) as txn:
            bin_file = txn.get(self.keys[index])

        image, target = loads(bin_file)

        if apply_transform:
            if self.is_image:
                image = self._decode_from_image(image)
            else:
                image = self._decode_from_array(image)

            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.root + ")"

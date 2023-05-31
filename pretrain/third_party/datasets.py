# A collection of datasets class used in pretraining models.

# imports.
import torch


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, aug, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.aug = aug

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.aug(x, self.preprocess), y
        else:
            im_tuple = (
                self.preprocess(x),
                self.aug(x, self.preprocess),
                self.aug(x, self.preprocess),
            )
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

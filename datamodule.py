import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from copy import deepcopy

from dataset import CT_Dataset


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class CT_Datamodule(LightningDataModule):
    def __init__(
        self,
        path: str,
        num_workers: int = 16,
        batch_size: int = 32,
        k: int = 0,
        num_folds=10,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train, self.val = None, None
        self.path = path
        self.k = k
        self.num_folds = num_folds
        num_samples = [99, 362, 114, 75]
        self.num_samples = [sum(num_samples) // self.num_folds] * (self.num_folds - 1)
        self.num_samples += [sum(num_samples) - sum(self.num_samples)]

        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((300, 300)),
                    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        mean=[0.1737, 0.1737, 0.1737], std=[0.2584, 0.2584, 0.2584]
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.1737, 0.1737, 0.1737], std=[0.2584, 0.2584, 0.2584]
                    ),
                ]
            ),
        }
        self.setup_done = False

    def prepare_data(self):
        if not self.setup_done:
            self.full_data = random_split(
                CT_Dataset(
                    "Dataset",
                    self.transform["train"],
                ),
                self.num_samples,
            )

        self.setup_done = True

    def set_k(self, k):
        self.k = k
        self.setup("fit")

    def setup(self, stage: str):
        self.train = deepcopy(torch.utils.data.ConcatDataset(
            [x for i, x in enumerate(self.full_data) if i != self.k]
        ))
        self.val = deepcopy(self.full_data[self.k])
        self.val.transform = self.transform["test"]
        self.train.transform = self.transform["train"]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers != 0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers != 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers != 0,
        )

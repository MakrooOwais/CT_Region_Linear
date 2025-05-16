import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from copy import deepcopy
from dataset import CT_Dataset
import os
import numpy as np
from collections import defaultdict


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
        
        self.transform = {
            "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((300, 300)),
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.1737, 0.1737, 0.1737], std=[0.2584, 0.2584, 0.2584]
                ),
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.1737, 0.1737, 0.1737], std=[0.2584, 0.2584, 0.2584]
                ),
            ]),
        }
        
        self.setup_done = False

    def prepare_data(self):
        if not self.setup_done:
            # Create a dataset with the train transform initially
            self.full_dataset = CT_Dataset(self.path, self.transform["train"])
            
            # Group images by patient ID
            patient_to_indices = self._group_by_patient()
            
            # Create patient-level folds
            self.fold_indices = self._create_patient_folds(patient_to_indices)
            
            self.setup_done = True

    def _group_by_patient(self):
        """Group image indices by patient ID extracted from the file path"""
        patient_to_indices = defaultdict(list)
        
        for idx, (_, _, full_scan_path) in enumerate(self.full_dataset.images):
            # Extract patient ID from the path
            # Assuming path format: path/region/class/patientID_scanID.tiff
            filename = os.path.basename(full_scan_path)
            patient_id = filename.split('_Slice_')[0]  # Adjust this based on your actual filename format
            
            patient_to_indices[patient_id].append(idx)
            
        return patient_to_indices

    def _create_patient_folds(self, patient_to_indices):
        """Create folds ensuring patients don't overlap between folds"""
        # Get list of all patient IDs
        patient_ids = list(patient_to_indices.keys())
        
        # Shuffle patient IDs
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(patient_ids)
        
        # Split patient IDs into folds
        fold_size = len(patient_ids) // self.num_folds
        patient_folds = [patient_ids[i:i + fold_size] for i in range(0, len(patient_ids), fold_size)]
        
        # Adjust if we have more folds than needed
        while len(patient_folds) > self.num_folds:
            # Distribute the last fold's patients among other folds
            last_fold = patient_folds.pop()
            for i, patient_id in enumerate(last_fold):
                patient_folds[i % len(patient_folds)].append(patient_id)
        
        # Convert patient IDs to image indices for each fold
        fold_indices = []
        for fold_patients in patient_folds:
            indices = []
            for patient_id in fold_patients:
                indices.extend(patient_to_indices[patient_id])
            fold_indices.append(indices)
            
        return fold_indices

    def set_k(self, k):
        self.k = k
        self.setup("fit")

    def setup(self, stage: str):
        # Validation set is the k-th fold
        val_indices = self.fold_indices[self.k]
        
        # Training set is all other folds
        train_indices = []
        for i in range(len(self.fold_indices)):
            if i != self.k:
                train_indices.extend(self.fold_indices[i])
        
        # Create train and validation subsets
        self.train = Subset(self.full_dataset, train_indices)
        self.val = Subset(self.full_dataset, val_indices)
        
        # Apply appropriate transforms
        self.train.dataset = deepcopy(self.full_dataset)
        self.train.dataset.transform = self.transform["train"]
        
        self.val.dataset = deepcopy(self.full_dataset)
        self.val.dataset.transform = self.transform["test"]

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

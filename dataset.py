import numpy as np
import os

from skimage import io
from torch.utils.data import Dataset


class CT_Dataset(Dataset):
    region2idx = {
        "Abdomen": 0,
        "Chest": 1,
        "Head_Neck": 2,
    }
    tumor2idx = {"Kinase": 0, "SDHB": 1, "SDHx": 1, "Sporadic": 2, "VHL_EPAS1": 3}

    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        regions = os.listdir(path)
        data = list()
        for region in regions:
            classes = os.listdir(os.path.join(path, region))
            for class_ in classes:
                scans = [
                    x
                    for x in os.listdir(os.path.join(path, region, class_))
                    if x.endswith("tiff")
                ]

                data.extend(
                    [
                        (class_, region, os.path.join(path, region, class_, scan))
                        for scan in scans
                    ]
                )
            self.images = data

    def __getitem__(self, idx):
        tumor, region, full_scan_path = self.images[idx]
        img = io.imread(full_scan_path)[..., 0:-1]

        if self.transform:
            img = self.transform(img)
        encoded_arr_region = np.zeros(4, dtype=float)
        encoded_arr_region[self.region2idx[region]] = 1
        encoded_arr_tumor = np.zeros(4, dtype=float)
        encoded_arr_tumor[self.tumor2idx[tumor]] = 1

        return encoded_arr_region, encoded_arr_tumor, img, full_scan_path

    def __len__(self):
        return len(self.images)

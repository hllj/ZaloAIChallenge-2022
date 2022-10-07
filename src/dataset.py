from pathlib import Path

import pandas as pd
import torchvision.transforms as tvf
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DehazeDataset(Dataset):
    def __init__(
        self, data_list, data_dir, unpaired=True, augment=True, crop_size=256
    ) -> None:
        super().__init__()

        self.data_list = pd.read_csv(data_list)
        self.data_dir = Path(data_dir)
        self.unpaired = unpaired
        self.input_size = crop_size

        self.transforms = []
        self.transforms += (
            [tvf.RandomCrop((self.input_size, self.input_size))]
            if crop_size != -1
            else [tvf.Pad(10, 10)]
        )
        self.transforms += [tvf.RandomHorizontalFlip()] if augment else []
        self.transforms += [tvf.ToTensor()]
        self.transforms = tvf.Compose(self.transforms)

    def __getitem__(self, index):
        if self.unpaired:
            clean_path = self.data_list.clean[index]
            hazy_path = self.data_list.hazy.sample(n=1).iloc[0]
        else:
            clean_path, hazy_path = self.data_list.iloc[index]

        clean = self.transforms(Image.open(self.data_dir / clean_path))
        hazy = self.transforms(Image.open(self.data_dir / hazy_path))

        return clean, hazy

    def __len__(self):
        return len(self.data_list)


class DehazeDatamodule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage=None) -> None:
        self.train_dataset = DehazeDataset(
            self.config.train_list,
            self.config.data_dir,
            unpaired=True,
            augment=True,
            crop_size=self.config.crop_size,
        )
        self.val_dataset = DehazeDataset(
            self.config.val_list,
            self.config.data_dir,
            unpaired=False,
            augment=False,
            crop_size=-1,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

from pathlib import Path
import os

import torch
import pandas as pd
import torchvision.transforms as tvf
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class LivenessDataset(Dataset):
    def __init__(
        self, data_list, data_dir, augment=True, crop_size=256
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        df = pd.read_csv(data_list)
        self.paths = df.iloc[:, 0]
        self.labels = list(map(str, df.iloc[:, 1]))
        
        self.input_size = crop_size
        self.transforms = get_image_transforms(self.input_size, augment)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.paths[index])
        image = Image.open(img_path)
        if self.transforms != None:
            image = self.transforms(image)

        label = torch.tensor(int(self.labels[index]))
        return image, label

    def __len__(self):
        return len(self.labels)


class LivenessDatamodule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        print('check', config['train_list'])
        self.config = config

    def setup(self, stage=None) -> None:
        self.train_dataset = LivenessDataset(
            self.config.train_list,
            self.config.data_dir,
            augment=True,
            crop_size=self.config.crop_size,
        )
        self.val_dataset = LivenessDataset(
            self.config.val_list,
            self.config.data_dir,
            augment=False,
            crop_size=self.config.crop_size,
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

def get_image_transforms(input_size, augment):
    transforms = [
        tvf.Resize([input_size, input_size])
    ]
    if augment:
        transforms += [
            tvf.RandomHorizontalFlip()
        ]
    transforms += [
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transforms = tvf.Compose(transforms)
    return transforms


# import hydra
# from omegaconf import DictConfig, OmegaConf
# @hydra.main(config_path="configs", config_name="default")
# def main(config: DictConfig) -> None:
#     print(config.dataset)
#     datamodule = LivenessDatamodule(config.dataset)
#     datamodule.setup()
#     train_loader = datamodule.train_dataloader()
#     batch = next(iter(train_loader))
#     print(batch[0])
#     print(batch[1])

# if __name__ == "__main__":
#     main()
import torch
import random
import numpy as np
import hydra
import pandas as pd
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as tvf

### seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark= False

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("mul", lambda x,y: int(float(x) * float(y)))
OmegaConf.register_new_resolver("as_list", lambda x,y: [x, y])

def get_image_transforms(input_size, augment_config=None, is_aug=False):
    transforms = []
    if is_aug:
        # transforms += [tvf.Resize([input_size, input_size])]
        for aug in augment_config:
            transforms += [hydra.utils.instantiate(augment_config[aug])]
    else:
        transforms += [
            tvf.Resize([input_size, input_size]),
        ]
    # transforms = tvf.RandomApply(torch.nn.ModuleList(transforms), p=0.6)
    transforms = tvf.Compose(transforms)
    return transforms

def transforms_func(image, config):
    transforms = get_image_transforms(config.dataset.crop_size, config.dataset.augmentation, config.dataset.is_aug)
    transforms_ori = get_image_transforms(config.dataset.crop_size, config.dataset.augmentation, False)
    width, height = image.size
    resize_transforms = tvf.Compose([
        tvf.CenterCrop((int(0.75 * height), int(0.75 * width))),
        tvf.Resize((int(0.375 * height), int(0.375 * width)))
    ])
    image = resize_transforms(image)
    if transforms is not None:
        image_tr = transforms(image)
    if transforms_ori is not None:
        image_ori = transforms_ori(image)
    return image_ori, image_tr

def aug(config):
    if not os.path.exists(config.dataset.save_dir):
        os.mkdir(config.dataset.save_dir)

    data_dir = config.dataset.data_dir
    df = pd.read_csv(config.dataset.data_list)
    paths = df.iloc[:, 0]
    labels = list(map(str, df.iloc[:, 1]))
    new_paths = []
    new_labels = []
    
    for path, label in zip(paths, labels):
        img_path = os.path.join(data_dir, path)
        print(img_path)
        image = Image.open(img_path)
        image_ori, image = transforms_func(image, config)
        save_path = os.path.join(config.dataset.save_dir, path)
        
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        image.save(save_path)

        filename, file_extension = os.path.splitext(path)
        path_ori = filename + '_ori' + file_extension
        save_path_ori = os.path.join(config.dataset.save_dir, path_ori)
        image_ori.save(save_path_ori)
        path = os.path.join(os.path.basename(config.dataset.save_dir), path)
        path_ori = os.path.join(os.path.basename(config.dataset.save_dir), path_ori)
        new_paths.extend([path,path_ori])
        new_labels.extend([label,label])
    
    out_df = pd.DataFrame(
    {'fname': new_paths,
     'liveness_score': new_labels
    })
    out_df.to_csv(config.dataset.save_list, index=False)

@hydra.main(config_path="configs", config_name="aug")
def main(config: DictConfig) -> None:
    print("Zalo AI Challenge - Liveness Detection")
    print(f"Current working directory : {Path.cwd()}")
    
    aug(config)


if __name__ == "__main__":
    main()
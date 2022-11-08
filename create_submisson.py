import argparse
import os
from glob import glob

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import get_image_transforms
from src.model import TIMMModel

from hydra import compose, initialize
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create submission")
    parser.add_argument("-ckpt", "--checkpoint", required=True)
    parser.add_argument("-cfg", "--config", required=True)
    args = parser.parse_args()


    cfg = OmegaConf.load(args.config)
    checkpoint_path = args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path)

    model = TIMMModel(cfg.model)
    # model.load_from_checkpoint(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)
    transforms = get_image_transforms(cfg.dataset.crop_size, False)

    submission_file = open("submission.csv", "w")
    submission_file.write("fname,liveness_score\n")
    list_folder = sorted(glob("data/public/images/*"))
    for folder in list_folder:
        name = folder.split("/")[-1]
        list_filename = os.listdir(folder)
        preds_list = []
        for filename in list_filename:
            path = os.path.join(folder, filename)
            image = Image.open(path)
            image = transforms(image)
            image.unsqueeze_(dim=0)
            image = image.to(device)
            logits = model(image)
            logits = F.softmax(logits, dim=-1)
            # preds = torch.argmax(logits, dim=1).item()
            preds = logits[0][1].item()
            print("output", path, preds)
            preds_list.append(preds)
        outputs = sum(preds_list) / len(preds_list)
        submission_file.write(f"{name + '.mp4'},{outputs}\n")
    submission_file.close()

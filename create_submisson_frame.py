import logging
import os
from glob import glob

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.dataset import get_image_transforms
from src.model import TIMMModel

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="inference")
def main(infer_config: DictConfig) -> None:
    checkpoint_path = infer_config.checkpoint

    output_folder = "/".join(checkpoint_path.split("/")[:-2])
    cfg = OmegaConf.load(
        os.path.join(infer_config.work_dir, output_folder, ".hydra", "config.yaml")
    )
    model = TIMMModel(cfg.model)
    ckpt = torch.load(os.path.join(infer_config.work_dir, checkpoint_path))
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    transforms = get_image_transforms(infer_config.crop_size, False, False)
    submission_file = open("submission_frame.csv", "w")
    submission_file.write("fname,liveness_score\n")
    list_folder = sorted(
        glob(os.path.join(infer_config.work_dir, "data/public/images/*"))
    )
    frame_idx = 0
    for folder in list_folder:
        name = folder.split("/")[-1]
        filename = sorted(os.listdir(folder))[frame_idx]
        preds_list = []
        path = os.path.join(folder, filename)
        image = Image.open(path)
        image = transforms(image)
        image.unsqueeze_(dim=0)
        image = image.to(device)
        logits = model(image)
        logits = F.softmax(logits, dim=-1)
        preds = logits[:, 1].item()
        outputs = preds
        submission_file.write(f"{name + '.mp4'},{outputs}\n")
    submission_file.close()


if __name__ == "__main__":
    main()

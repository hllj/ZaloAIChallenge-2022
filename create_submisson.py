import logging
import os
from glob import glob

import cv2
import hydra
import numpy as np
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
    transforms = get_image_transforms(infer_config.crop_size, False, None)
    submission_file = open("submission.csv", "w")
    submission_file.write("fname,liveness_score\n")
    list_folder = sorted(
        glob(os.path.join(infer_config.work_dir, "data/public/pil_images/*"))
    )
    for folder in list_folder:
        name = folder.split("/")[-1]
        list_filename = sorted(os.listdir(folder))
        preds_list = []
        for idx, filename in enumerate(list_filename):
            path = os.path.join(folder, filename)
            image = Image.open(path)
            image = transforms(image)
            image.unsqueeze_(dim=0)
            image = image.to(device)
            logits = model(image)
            logits = F.softmax(logits, dim=-1)
            preds = logits[:, 1].item()
            log.info(f"{path} {idx}: {preds}")
            preds_list.append(preds)
            # only extract frame 0
            break
        outputs = sum(preds_list) / len(preds_list)
        submission_file.write(f"{name + '.mp4'},{outputs}\n")
    submission_file.close()

    # list_video = sorted(
    #     glob(os.path.join(infer_config.work_dir, "data/public/videos/*"))
    # )
    # for video_path in list_video:
    #     name = video_path.split('/')[-1]
    #     cap = cv2.VideoCapture(video_path)
    #     frames = []
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     if not fps:
    #         fps = 25
    #     count = 0
    #     preds_list = []
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if isinstance(frame, np.ndarray):
    #             if int(count % round(fps)) == 0:
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 image = Image.fromarray(frame)
    #                 image = transforms(image)
    #                 image.unsqueeze_(dim=0)
    #                 image = image.to(device)
    #                 logits = model(image)
    #                 logits = F.softmax(logits, dim=-1)
    #                 preds = logits[:, 1].item()
    #                 log.info(f"{video_path} {count}: {preds}")
    #                 preds_list.append(preds)
    #             count += 1
    #             break
    #         else:
    #             break
    #     outputs = sum(preds_list) / len(preds_list)
    #     submission_file.write(f"{name},{outputs}\n")
    # submission_file.close()


if __name__ == "__main__":
    main()

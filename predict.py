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
from tqdm import tqdm
from src.dataset import get_image_transforms
from src.model import TIMMModel

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="inference")
def main(infer_config: DictConfig) -> None:

    def get_model(infer_config, checkpoint):
        checkpoint_path = checkpoint

        output_folder = "/".join(checkpoint_path.split("/")[:-2])
        cfg = OmegaConf.load(
            os.path.join(infer_config.work_dir, output_folder, ".hydra", "config.yaml")
        )
        model = TIMMModel(cfg.model)

        ckpt = torch.load(os.path.join(infer_config.work_dir, checkpoint_path))
        model.load_state_dict(ckpt["state_dict"])

        return model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_s = get_model(infer_config, infer_config.checkpoint_s).eval().to(device)
    model_h = get_model(infer_config, infer_config.checkpoint_h).eval().to(device)

    transforms = get_image_transforms(infer_config.crop_size, False, None)

    # predict
    submission_file = open("submission.csv", "w")
    submission_file.write("fname,liveness_score\n")
    list_video = sorted(
        glob(infer_config.videos_dir)
    )
    for video_path in list_video:
        name = video_path.split('/')[-1]
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps:
            fps = 25
        count = 0

        images = []
        while cap.isOpened():
            ret, frame = cap.read()
            if isinstance(frame, np.ndarray):
                if int(count % round(fps)) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    image = transforms(image)
                    images.append(image)
                    
                    
                count += 1
            else:
                break
        images = torch.stack(images)
        images = images.to(device)

        logits_s = model_s(images)
        logits_s = F.softmax(logits_s, dim=-1)
        outputs_s = logits_s[:, 1].mean().item()

        logits_h = model_h(images)
        logits_h = F.softmax(logits_h, dim=-1)
        outputs_h = logits_h[:, 1].mean().item()

        outputs = (outputs_s + outputs_h)/2

        log.info(f"{video_path}: {outputs}")
        submission_file.write(f"{name},{outputs}\n")
    submission_file.close()

    
if __name__ == "__main__":
    main()

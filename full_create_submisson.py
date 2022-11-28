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
gt = {"0": 1,"100": 1,"1001": 1,"1005": 1,"1022": 1,"1023": 1,"1045": 0,"1048": 1,"1049": 0,"1062": 0,"1079": 1,"1092": 1,"1102": 0,"1116": 1,"1122": 1,"1132": 0,"1133": 0,"1141": 0,"1151": 1,"1155": 0,"1156": 0,"1159": 0,"1178": 1,"118": 1,"1180": 0,"1187": 1,"1201": 1,"1208": 1,"121": 1,"1214": 1,"1220": 0,"1224": 0,"1227": 0,"123": 1,"1239": 1,"1243": 1,"1246": 0,"1247": 1,"1249": 1,"1255": 0,"1258": 0,"1259": 1,"1261": 0,"1266": 1,"1268": 1,"1271": 1,"1272": 0,"1276": 0,"1282": 0,"1284": 1,"1287": 1,"1291": 0,"1295": 0,"1298": 0,"1299": 1,"1305": 0,"1310": 0,"1311": 0,"1316": 0,"132": 1,"1327": 0,"1332": 0,"1334": 0,"1335": 1,"1338": 1,"1340": 0,"1346": 1,"1348": 1,"1355": 0,"1379": 0,"1387": 0,"139": 1,"1391": 0,"1392": 0,"1393": 1,"1402": 1,"1405": 1,"1408": 0,"1414": 1,"1426": 0,"1429": 1,"1439": 1,"1440": 0,"1442": 1,"1451": 0,"1455": 0,"1462": 1,"1471": 0,"1474": 1,"1481": 0,"149": 1,"1495": 0,"1496": 0,"1499": 0,"15": 1,"1502": 0,"1506": 1,"1511": 0,"1512": 1,"1514": 0,"1517": 0,"1519": 1,"1525": 0,"1527": 0,"153": 0,"1530": 0,"1536": 1,"1537": 0,"1545": 0,"1548": 1,"1558": 1,"1560": 0,"1561": 1,"1563": 0,"1566": 0,"1571": 1,"158": 0,"1586": 0,"1589": 1,"159": 0,"1595": 1,"160": 0,"1600": 0,"1613": 1,"1616": 0,"162": 0,"1631": 0,"1632": 1,"1634": 0,"1638": 0,"1643": 1,"1648": 1,"1662": 1,"1673": 1,"1677": 0,"1678": 1,"1682": 1,"1685": 0,"1686": 1,"169": 1,"17": 0,"1700": 0,"1708": 1,"1709": 1,"1723": 0,"174": 0,"1744": 0,"1745": 1,"1747": 1,"1753": 1,"1755": 1,"1759": 0,"1764": 0,"178": 0,"1780": 0,"1784": 0,"1792": 1,"1793": 0,"1798": 0,"1800": 1,"1806": 0,"1819": 0,"1830": 1,"1836": 1,"184": 1,"1846": 0,"1847": 0,"1853": 1,"186": 1,"1861": 0,"1864": 1,"1867": 1,"187": 0,"1878": 0,"188": 1,"1883": 1,"1885": 0,"190": 0,"201": 0,"202": 1,"206": 1,"21": 0,"212": 0,"214": 1,"215": 1,"224": 1,"226": 0,"228": 1,"23": 1,"234": 0,"235": 1,"237": 0,"250": 1,"266": 0,"271": 0,"278": 0,"286": 0,"289": 1,"293": 0,"297": 0,"300": 1,"302": 0,"303": 1,"311": 0,"315": 0,"318": 1,"320": 0,"330": 0,"338": 1,"34": 0,"342": 1,"358": 0,"364": 1,"368": 1,"370": 0,"377": 0,"378": 0,"38": 0,"381": 0,"387": 0,"397": 1,"4": 1,"400": 1,"404": 0,"406": 0,"410": 1,"411": 0,"412": 0,"437": 0,"450": 1,"460": 1,"466": 0,"47": 0,"474": 1,"479": 1,"482": 0,"486": 1,"488": 1,"49": 1,"496": 1,"500": 0,"507": 1,"51": 1,"510": 0,"516": 1,"522": 0,"523": 1,"527": 1,"528": 1,"537": 0,"539": 1,"542": 1,"545": 1,"552": 1,"566": 1,"567": 0,"569": 0,"57": 1,"574": 0,"578": 1,"581": 0,"586": 1,"592": 1,"60": 1,"607": 1,"616": 1,"62": 0,"621": 0,"626": 1,"634": 1,"638": 0,"64": 1,"647": 0,"648": 0,"659": 1,"660": 0,"661": 0,"666": 0,"672": 1,"677": 0,"680": 0,"683": 0,"689": 0,"696": 1,"703": 0,"705": 0,"707": 1,"708": 0,"714": 0,"717": 1,"722": 1,"724": 1,"731": 1,"733": 1,"740": 1,"748": 1,"757": 0,"760": 1,"762": 0,"766": 1,"77": 1,"770": 1,"773": 0,"776": 1,"777": 1,"786": 0,"790": 1,"794": 0,"799": 0,"808": 1,"828": 1,"830": 1,"832": 0,"841": 0,"843": 1,"853": 0,"857": 1,"861": 1,"867": 1,"872": 1,"876": 0,"879": 0,"880": 0,"881": 1,"882": 1,"884": 0,"891": 1,"892": 1,"901": 1,"902": 0,"916": 1,"917": 1,"919": 1,"925": 0,"933": 1,"935": 1,"944": 1,"95": 1,"953": 0,"957": 1,"959": 0,"960": 0,"962": 1,"965": 0,"966": 1,"972": 0,"973": 1,"979": 1,"981": 1,"983": 1}
checkpoint_path = "multirun/baseline-tf_efficientnet_b4_ns/2022-11-10-13-58-38/ckpts/last.ckpt"

def get_ckpt(runtime_type="multirun", specific_archit="",month_day=[],to_hour=[]):
    all_ckpts = []
    if specific_archit:
        model_dates = sorted(glob(f"{runtime_type}/{specific_archit}/*"))
        for model_date in model_dates:
            create_day = model_date.split("/")[2]
            ckpts = []
            if not month_day:
                ckpts = sorted(glob(f"{model_date}/ckpts/*"))
            elif create_day[8:10] in month_day:
                if not to_hour:
                    ckpts = sorted(glob(f"{model_date}/ckpts/*"))
                elif to_hour[0] <= int(create_day[11:13]) <= to_hour[1]:
                    ckpts = sorted(glob(f"{model_date}/ckpts/*"))
            all_ckpts.extend(ckpts)
    else:
        architectures = [d for d in os.listdir(runtime_type) if d[:5] != "infer" and d[-3:] != "csv"]
        for archit in architectures:
            model_dates = sorted(glob(f"{runtime_type}/{archit}/*"))
            for model_date in model_dates:
                ckpts = []
                create_day = model_date.split("/")[2]
                # print(create_day, month_day)
                if not month_day:
                    ckpts = sorted(glob(f"{model_date}/ckpts/*"))
                elif create_day[8:10] in month_day:
                    if not to_hour:
                        ckpts = sorted(glob(f"{model_date}/ckpts/*"))
                    elif to_hour[0] <= int(create_day[11:13]) <= to_hour[1]:
                        ckpts = sorted(glob(f"{model_date}/ckpts/*"))
                all_ckpts.extend(ckpts)
    return all_ckpts
def test_from_checkpoint(checkpoint_path) -> None:
    model_tags = checkpoint_path.split("/")
    runtime_type = model_tags[0]
    summary_file = f"{runtime_type}/summary.csv"
    model_folder = "/".join(model_tags[:-2])
    cfg = OmegaConf.load(os.path.join(model_folder, ".hydra/config.yaml"))
    if "loss" not in cfg.model:
        cfg.model["loss"] = {"_target_":"torch.nn.CrossEntropyLoss", "label_smoothing": "0.0"}
        # loss:
        # _target_: torch.nn.CrossEntropyLoss
        # label_smoothing: 0.0
    model = TIMMModel(cfg.model)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    transforms = get_image_transforms(cfg.dataset.crop_size, False, None)

    ### output
    model_name = '-'.join(model_tags[1].split('-')[1:])
    model_day = model_tags[2]
    output_folder = os.path.join(f"{runtime_type}/inference_{model_name}",model_day)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # print("out folder", output_folder)
    false_case = 0
    with open(output_folder+"/submission.csv", "w") as submission_file:# create in infer folder
        submission_file.write("fname,liveness_score\n")
        list_folder = sorted(glob("data/public/pil_images/*"))
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
            round_outputs = round(outputs)
            if gt[name] != round_outputs:
                false_case += 1
            submission_file.write(f"{name + '.mp4'},{outputs}\n")
        # print(f"false_case: {false_case}")
    with open(summary_file, 'a') as fa:
        fa.write("{},{}\n".format(checkpoint_path,false_case))
    # submission_file.close()

if __name__ == "__main__":
    # test_from_checkpoint()
    ckpts = get_ckpt(month_day=["13"], to_hour=[])
    for ckpt in tqdm(ckpts):
        test_from_checkpoint(ckpt)
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from utils import Dice_score_Binary, Dice_score_multi
from oxford_pet import OxfordPetDataset
from models.unet import Unet
from utils import make_submission

data_dir  = Path(r"/home/re6141029/DL/dl_lab2/dataset/oxford-iiit-pet")
ckpt_dir  = Path(r"/home/re6141029/DL/dl_lab2/saved_models/unet/best_unet/best_unet.pth")
save_infer_dir  = Path(r"/home/re6141029/DL/dl_lab2/experiment/unet")
save_infer_dir.mkdir(parents=True, exist_ok=True)
save_submission = Path(r"/home/re6141029/DL/dl_lab2/submisson/submission.csv")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch   = 1
mode    = "Multi"
model   = Unet(1,2).to(device)

test_data = OxfordPetDataset(data_dir, "test")
test      = DataLoader(test_data, batch, False)
ckpt      = torch.load(ckpt_dir, device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    bar = tqdm(test, desc="Inference")

    for batch in bar:
        imgs = batch["image"].to(device)
        ids  = batch["id"]

        output  = model(imgs)
        preds   = torch.argmax(output, dim=1)
        preds   = preds.cpu().np().astype(np.int8)
        
        for preds, ids in zip(preds, ids):
            seg_image = Image.fromarray(preds)
            seg_image.save(save_infer_dir / f"{ids}.png")
print(f"Saved unet inference images to: {save_infer_dir}")

make_submission(save_infer_dir, save_submission)
print(f"Saved submission to: {save_submission}")




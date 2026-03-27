import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


#Hyperparameters
threshold = 0.5

#for val scoring   
def Dice_score_Binary(logits, targets, threshold=threshold, eps=1e-8):
    probs       = torch.sigmoid(logits)
    preds       = (probs > threshold).float()
    targets     = targets.float()

    intersec    = (preds*targets).sum(dim=(1,2,3))
    union       = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice_score  = 2*intersec / (union+eps) #avoid divide zero
    return dice_score.mean().item()

#for training loss
def Dice_score_Binary_train(logits, targets, eps=1e-8):
    probs       = torch.sigmoid(logits)
    targets     = targets.float()

    intersec    = (probs*targets).sum(dim=(1,2,3))
    union       = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice_score  = 2*intersec / (union+eps) 
    return 1-dice_score.mean()

class Dice_loss_binary(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = Dice_score_Binary_train()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

#for val scoring    
def Dice_score_multi(logits, targets, fg=1, eps=1e-8):
    preds       = torch.argmax(logits, dim=1)
    preds       = (preds == fg).float()
    targets     = (targets == fg).float()

    intersec    = (preds*targets).sum(dim=(1,2)) 
    union       = preds.sum(dim=(1,2)) + targets.sum(dim=(1,2))
    dice_score  = 2*intersec / (union+eps)
    return dice_score.sum().item()

#for training loss
def Dice_score_multi_train(logits, targets, fg=1, eps=1e-8):
    preds       = torch.softmax(logits, dim=1)
    preds       = preds[:, fg, :, :]
    targets     = (targets == fg).float()

    intersec    = (preds*targets).sum(dim=(1,2)) 
    union       = preds.sum(dim=(1,2)) + targets.sum(dim=(1,2))
    dice_score  = 2*intersec / (union+eps)
    return 1 - dice_score.mean()

class Dice_loss_multi(nn.Module):
    def __init__(self, ce_weight, dice_weight, fg=1):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.fg          = fg

    def forward(self, logits, targets):
        ce_loss         = self.ce(logits, targets.long())
        dice_multi_loss = Dice_score_multi_train(logits, targets, self.fg)
        multi_loss      = self.ce_weight*ce_loss + self.dice_weight*dice_multi_loss    
        return multi_loss
    
#for submission
def rle_encode(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]]) # zero padding, help recognite value change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 # with correct transfer 
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def make_submission(pred_dir, output_csv):
    pred_dir = Path(pred_dir)
    output_csv = Path(output_csv)

    rows = []
    for png_path in sorted(pred_dir.glob("*.png")):
        image_id = png_path.stem
        mask = np.array(Image.open(png_path))
        mask = (mask > 0).astype(np.uint8)

        encoded_mask = rle_encode(mask)
        rows.append({
            "image_id": image_id,
            "encoded_mask": encoded_mask
        })

    df = pd.DataFrame(rows, columns=["image_id", "encoded_mask"])
    df.to_csv(output_csv, index=False)
    return df

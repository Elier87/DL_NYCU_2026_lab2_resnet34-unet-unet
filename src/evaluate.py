import torch
from utils import Dice_score_Binary, Dice_score_multi

def evaluate(model, val_data, criterion, device, mode="Multi"):
    model.eval()
    cum_loss = 0
    cum_dice = 0

    with torch.no_grad():
        for batch in val_data:
            imgs    = batch["image"].to(device)
            if mode == "multi": 
                masks   = batch["mask"].long().to(device)
            else:
                masks   = batch["mask"].float().to(device)
            
            output  = model(imgs)
            loss    = criterion(output, masks)
            cum_loss += loss.item()

            if mode == "multi":
                cum_dice += Dice_score_multi(output, masks)
            else:
                cum_dice += Dice_score_Binary(output, masks)
    
    avg_val_loss    = cum_loss/len(val_data)
    avg_val_dice    = cum_dice/len(val_data)
    return avg_val_loss, avg_val_dice

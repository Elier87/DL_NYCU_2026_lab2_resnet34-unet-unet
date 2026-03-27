import torch
import torch.nn as nn 
import torch.optim as optim
from oxford_pet import OxfordPetDataset
from models.unet import Unet
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import evaluate
from utils import Dice_loss_multi, Dice_loss_binary

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = Path(r"/home/re6141029/DL/dl_lab2/dataset/oxford-iiit-pet")
save_dir = Path(r"/home/re6141029/DL/dl_lab2/saved_models/unet")

lr       = 1e-3
epochs   = 1
momentum = 0.99
batch    = 1
mode     = "Multi"
dice_weight = 0.5

model       = Unet(1,2).to(device)

if mode == "Multi":
    criterion = Dice_loss_multi(dice_weight, 1-dice_weight)
else:
    criterion = Dice_loss_binary(dice_weight, 1-dice_weight) 
optimizer   = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_data  = OxfordPetDataset(data_dir, "train")
val_data    = OxfordPetDataset(data_dir, "val")
train       = DataLoader(train_data, batch, False)
val         = DataLoader(val_data, batch, False)

best_model_loss  = float("inf")
best_model       = ""
best_dice_score  = -1.0
history     = {
    "train_loss" : [],
    "val_loss"   : [],
    "val_dice"   :[]
}

for epoch in range(epochs):
    model.train()
    cum_loss    = 0.0
    bar         = tqdm(train, desc=f"Epoch: {epoch+1}/{epochs}")
    
    for batch in bar:  
        imgs        = batch["image"].to(device)
        masks       = batch["mask"].to(device)

        optimizer.zero_grad()    
        outputs     = model(imgs)
        loss = criterion(imgs, masks)
        loss.backward()
        optimizer.step()
        cum_loss    += loss.item()
                
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")
    
    avg_train_loss      = cum_loss/len(train)

    avg_val_loss, avg_val_dice = evaluate(model, val, criterion, device, mode)
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_dice"].append(avg_val_dice)

    ckpt    = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_dice": avg_val_dice,
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "val_dice_history": history["val_dice"],
        "lr": lr
    }

    #torch.save(ckpt, save_dir/"latest.pth")

    save_path = save_dir / f"unet_{epoch+1:02d}_{avg_train_loss:.3f}_{avg_val_loss:.3f}.pth"
    torch.save(ckpt, save_path)

    if avg_val_loss < best_model_loss:
        best_model = ckpt
        best_path  = save_dir / "best_unet" / f"unet_{epoch+1:02d}_{avg_train_loss:.3f}_{avg_val_loss:.3f}.pth"

    print(f"Epoch: {epoch+1}/{epochs}, avg_train_loss: {avg_train_loss:.4f}, avg_val_loss: {avg_val_loss:.4f}")

torch.save(ckpt, best_path)    
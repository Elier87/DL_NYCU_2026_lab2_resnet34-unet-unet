import torch
import torch.nn as nn

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
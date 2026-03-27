import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

data_dir = Path(r"/home/re6141029/DL/dl_lab2/dataset/oxford-iiit-pet/")

class OxfordPetDataset(Dataset):
    def __init__(self, root, split="train", size=(572 ,572), transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.images_dir = self.root / "images"
        self.mask_dir = self.root / "annotations" / "trimaps"
        self.split_dir = self.root / "splits"

        if self.split == "train":
            split_file = self.split_dir / "train.txt"
        elif self.split == "val":
            split_file = self.split_dir / "val.txt"
        elif self.split == "test":
            split_file = self.split_dir / "test_unet.txt"
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.image_ids      = self._read_split_txt(split_file)
        self.to_tensor      = T.ToTensor()
        self.img_size       = size
        self.image_resize   = T.Resize(size, interpolation=InterpolationMode.BILINEAR)
        self.mask_resize    = T.Resize((388, 388), interpolation=InterpolationMode.NEAREST)
        

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self._get_image_path(image_id)
        image = Image.open(image_path).convert("L")
        image = self.image_resize(self.to_tensor(image))

        if self.split != "test":
            mask_path = self._get_mask_path(image_id)
            trimap = np.array(Image.open(mask_path))
            mask = (trimap == 1).astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = self.mask_resize(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            mask = torch.zeros((self.img_size[0],self.img_size[1]),dtype=torch.long)
        return {
            "image" : image,
            "mask"  : mask,
            "id"    : image_id  
        }

    
    def __len__(self):
        return len(self.image_ids)

    def _get_image_path(self, image_id):
        return self.images_dir / f"{image_id}.jpg"
    
    def _get_mask_path(self, image_id):
        return self.mask_dir / f"{image_id}.png"
     
    #從splits挑出train, val, test
    def _read_split_txt(self, split_file):
        with open(split_file,"r") as f:
            ids = f.readlines()
        image_ids = []
        for idx in ids:
            idx = idx.strip()
            if idx == "":
                continue
            image_ids.append(idx)
        return image_ids
        
        

data = OxfordPetDataset(data_dir, "train")
train = next(iter(DataLoader(data,80,None)))
print(train["image"][55])
print(train["image"].shape)
print(train["mask"][55])
print(train["id"][55])

    

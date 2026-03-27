import torch
import torch.nn as nn

class double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block  = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3),
            nn.ReLU()           
        )
    def forward(self, img):
        return self.block(img)

class down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = double_conv(in_c, out_c)  
        self.pool = nn.MaxPool2d(2)              

    def forward(self, img):
        img_conv = self.conv(img)       #for skip_channel
        img_pool = self.pool(img_conv)        
        return img_conv, img_pool
    

def copy_crop(src, target):
    _, _, h_s, _ = src.shape 
    _, _, h_t, _ = target.shape

    delta_ridge = h_s - h_t
    top     = delta_ridge // 2
    down    = top + h_t

    return src[:, :, top:down, top:down] #c'z square



class up(nn.Module):
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_c, out_c, 2, 2)
        self.conv   = double_conv(out_c+skip_c, out_c)

    def forward(self, img, img_skip):
        img      = self.up(img)
        img_skip = copy_crop(img_skip, img)
        img      = torch.cat([img_skip, img], dim = 1)
        img      = self.conv(img)
        return img

class outconv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
    
    def forward(self, img):
        return self.conv(img)


class Unet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down1 = down(n_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.bottom = double_conv(512, 1024)
        
        self.up1    = up(1024, 512, 512)
        self.up2    = up(512, 256, 256)
        self.up3    = up(256, 128, 128)
        self.up4    = up(128, 64, 64)

        self.out_c  = outconv(64, n_classes)

    def forward(self, img):
        skip1, img1 = self.down1(img)
        skip2, img2 = self.down2(img1)
        skip3, img3 = self.down3(img2)
        skip4, img4 = self.down4(img3)

        mid_img = self.bottom(img4)

        img5        = self.up1(mid_img, skip4)
        img6        = self.up2(img5, skip3)
        img7        = self.up3(img6, skip2)
        img8        = self.up4(img7, skip1)

        output      =self.out_c(img8)

        return output





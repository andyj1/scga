import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import resize, normalize

###########################
# Generator: Resnet
###########################

def batched_crop_based_on_mask(images, masks):
    """
    Crops a batch of image tensors based on a batch of mask tensors.

    Args:
        images (torch.Tensor): A batch of image tensors of shape (B, C, H, W).
        masks (torch.Tensor): A batch of mask tensors of shape (B, 1, H, W) or (B, H, W).

    Returns:
        list: A list of cropped image tensors. The shapes of the cropped images
              may vary depending on the content of the masks.
    """
    cropped_images = []
    for image, mask in zip(images, masks):
        # Ensure mask is 2D (H, W)
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        # Find the coordinates of the True values
        y_indices, x_indices = torch.where(mask)

        if y_indices.numel() == 0 or x_indices.numel() == 0:
            cropped_images.append(torch.empty(0, image.size(0), 0, 0)) # Empty tensor if no mask
            continue

        # Find the min and max coordinates
        min_y, max_y = torch.min(y_indices), torch.max(y_indices)
        min_x, max_x = torch.min(x_indices), torch.max(x_indices)

        # Crop the image
        cropped_image = image[:, min_y:max_y + 1, min_x:max_x + 1]
        cropped_images.append(cropped_image)

    return torch.stack(cropped_images)


class Denormalize(object):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)

    def __call__(self, tensor):
        """Denormalize a tensor image with mean and standard deviation."""
        device = tensor.device
        if tensor.ndim == 3:  # Handle single image tensors (C, H, W)
            tensor = tensor * self.std.to(device) + self.mean.to(device)
        elif tensor.ndim == 4: # Handle batches of images (B, C, H, W)
            tensor = tensor * self.std.to(device).unsqueeze(0) + self.mean.to(device).unsqueeze(0)
        return tensor

# To control feature map in generator



# weight init from LTP (GAPF)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class GeneratorResnet(nn.Module):
    def __init__(self, inception = False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        '''
        super(GeneratorResnet, self).__init__()
        self.ngf = 64  # Number of filters in first layer
        
        # Input_size = 3, n, n
        self.orig_w = 224
        self.orig_h = 224

        self.inception = inception
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, self.ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(self.ngf * 4)
        self.resblock2 = ResidualBlock(self.ngf * 4)
        self.resblock3 = ResidualBlock(self.ngf * 4)
        self.resblock4 = ResidualBlock(self.ngf * 4)
        self.resblock5 = ResidualBlock(self.ngf * 4)
        self.resblock6 = ResidualBlock(self.ngf * 4)


        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        self.denormalize = Denormalize()


    def preprocess_image(self, input_img, mask=None):
        # if mask is not None:
        #     input_img = batched_crop_based_on_mask(input_img,mask)
        input_img = self.denormalize(input_img)
        self.orig_h = input_img.shape[2]
        self.orig_w = input_img.shape[3]
        orig_image = input_img
        input_img = resize(input_img, (224, 224))
        return input_img, orig_image

    def postprocess_image(self, input_img, orig_img):
        input_img = resize(input_img, (self.orig_h, self.orig_w))

        eps=10/255.0
        adv = torch.min(torch.max(input_img, orig_img - eps), orig_img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        return adv
    
    def forward(self, input, mask=None, feat=False):
        x, orig_image = self.preprocess_image(input, mask=mask)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        feats = {}
        feats['block3'] = x.clone()  # input to resblock1 # 0
        x = self.resblock1(x)
        feats['resblock1'] = x.clone() # input to resblock2 / output of resblock1 # 1
        x = self.resblock2(x)
        feats['resblock2'] = x.clone() # input to resblock3 / output of resblock2 # 2
        x = self.resblock3(x)
        feats['resblock3'] = x.clone()
        x = self.resblock4(x)
        feats['resblock4'] = x.clone()
        x = self.resblock5(x)
        feats['resblock5'] = x.clone()
        x = self.resblock6(x)
        feats['resblock6'] = x.clone()

        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
    
        x = (torch.tanh(x) + 1) / 2 # Output range [0 1]
        
        x = self.postprocess_image(x, orig_image)
        
        if feat:
            return x, feats  # adv image + list of 6 feature maps
        else:
            return x

    def forward_features(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual



if __name__ == '__main__':
    import time 
    netG = GeneratorResnet()
    img = torch.randn(1, 3, 224, 224)
    start = time.time()
    perturbation = netG(img)
    end = time.time()
    print(perturbation.shape)  # Expected: (1, 3, 224, 224)
    print(f'Elapsed time: {end - start:.4f} seconds')

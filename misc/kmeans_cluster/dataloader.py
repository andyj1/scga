import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

def get_imagenet_s_subset_classes(data_root, subset_name='ImageNetS_categories_im50.txt'):
    """Reads the class ID list for a given ImageNet-S subset."""
    subset_file = os.path.join(data_root, '../..', 'data/categories', subset_name)
    if not os.path.exists(subset_file):
        raise FileNotFoundError(f"Subset class list not found at: {subset_file}")
    
    with open(subset_file, 'r') as f:
        class_ids = [line.split()[0] for line in f.read().splitlines()]
    return set(class_ids)

class ImageNetSSegmentationDataset(Dataset):
    """
    Custom Dataset for ImageNet-S to load images, segmentation masks, and image paths.
    It can be filtered to load a specific subset of classes (e.g., ImageNetS50).
    """
    def __init__(self, data_root, split='validation', subset_class_ids=None, transform=None, mask_transform=None):
        self.transform = transform
        self.mask_transform = mask_transform
        
        if split == 'validation':
            image_folder = os.path.join(data_root, 'ImageNetS919', 'validation')
            mask_folder = os.path.join(data_root, 'ImageNetS919', 'validation-segmentation')
        else:
            image_folder = os.path.join(data_root, 'ImageNetS919', split)
            mask_folder = os.path.join(data_root, 'ImageNetS919', f'{split}_segmentation')

        if not os.path.isdir(image_folder) or not os.path.isdir(mask_folder):
            raise FileNotFoundError(f"Image or mask directory not found for split '{split}'.")

        all_image_paths = sorted(glob.glob(os.path.join(image_folder, '**', '*.JPEG'), recursive=True))
        
        self.image_paths = []
        self.mask_paths = []

        print(f"Verifying image-mask pairs for split '{split}'...")
        for img_path in all_image_paths:
            class_id = os.path.basename(os.path.dirname(img_path))
            if subset_class_ids and class_id not in subset_class_ids:
                continue

            rel_path = os.path.relpath(img_path, image_folder)
            mask_path = os.path.join(mask_folder, os.path.splitext(rel_path)[0] + '.png')
            
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        
        print(f"Found {len(self.image_paths)} valid image-mask pairs for the specified subset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Get the class name from the directory structure
        class_name = os.path.basename(os.path.dirname(img_path))
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask, img_path, class_name

def get_dataloader(data_root, split='validation', subset='im50', batch_size=32, num_workers=4):
    subset_class_ids = None
    if subset == 'im50':
        subset_class_ids = get_imagenet_s_subset_classes(data_root, 'ImageNetS_categories_im50.txt')
    elif subset == 'im300':
        subset_class_ids = get_imagenet_s_subset_classes(data_root, 'ImageNetS_categories_im300.txt')

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    dataset = ImageNetSSegmentationDataset(
        data_root=data_root, 
        split=split, 
        subset_class_ids=subset_class_ids,
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    shuffle = (split == 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

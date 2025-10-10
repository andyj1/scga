import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFile
# from skimage import transform
# from torch.autograd import Variable
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms, utils
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, extract_archive
import torchvision.transforms as T

from tqdm import tqdm
import random
import glob

from rich import print

import warnings
warnings.filterwarnings("ignore")
# plt.ion()   # interactive mode

try:
    from datasets.DCL_finegrained import dataset, config, dataset_DCL
except:
    from DCL_finegrained import dataset, config, dataset_DCL # for datasets/train.py test

ImageFile.LOAD_TRUNCATED_IMAGES = False

class CarsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mat_anno, data_dir, car_names, train_transform=None, target_transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.train_transform = train_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])

        try:
            image = Image.open(img_name)
        except:
            # return None, None
            raise IOError(f'error reading image: {img_name}')
            
        car_class = self.car_annotations[idx][-2][0][0]

        if self.train_transform:
            image = self.train_transform(image)

        if self.target_transform is not None:
            image = self.target_transform(image)

        return image, car_class



class CUBDataset(Dataset):
    def __init__(self, root, train_transform=None, target_transform=None, loader=default_loader, train=True):
        super(CUBDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.loader = default_loader

        self.train = train
        self.train_transform = train_transform
        self.target_transform = target_transform

        self._load_metadata()
        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root,  'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        self._load_metadata()
        # try:
        #     self._load_metadata()
        # except Exception:
        #     return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, 'images', row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'images', sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        # print(img)
        # sys.exit(1)

        if self.train_transform is not None:
            img = self.train_transform(sample)

        if self.target_transform is not None:
            img = self.target_transform(img)

        return img, target


class AircraftDataset(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', train_transform=None, target_transform=None, download=False):
        super(AircraftDataset, self).__init__(root)
        self.root = root

        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train_transform = train_transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.train_transform is not None:
            sample = self.train_transform(sample)

        if self.target_transform is not None:
            sample = self.target_transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


class PaintingsDataset(Dataset):
    """Face Landmarks dataset.
        WikiArtDataset
        """
    def __init__(self, csv_file, root_dir, train_transform=None, target_transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.artist_frame = pd.read_csv(csv_file)
        self.artist_frame = self.artist_frame.loc[self.artist_frame['in_train']==train].reset_index(drop=True)
        # Remove the train and test, vs train only field, and the is training field.
        self.artist_frame = self.artist_frame.drop(['artist_group', 'in_train'], axis=1)

        # Pick a threshold that makes 10 categories of genre
        genre_threshold = 1600

        value_counts = self.artist_frame['genre'].value_counts() # Specific column
        to_remove = value_counts[value_counts <= genre_threshold].index
        self.artist_frame['genre'].replace(to_remove, 'other', inplace=True)
        self.artist_frame = self.artist_frame.reset_index(drop=True)
        # Pick a threshold that makes 10 categories of style
        style_threshold = 300

        value_counts = self.artist_frame['style'].value_counts() # Specific column
        to_remove = value_counts[value_counts <= style_threshold].index
        self.artist_frame['style'].replace(to_remove, 'other', inplace=True)
        self.artist_frame = self.artist_frame.reset_index(drop=True)
        index_names = self.artist_frame[self.artist_frame['pixelsx'] >= 18000].index
        self.artist_frame.drop(index_names, inplace = True)
        self.artist_frame = self.artist_frame.reset_index(drop=True)

        # Keep only one entry per file
        self.artist_frame = self.artist_frame.drop_duplicates(subset=['new_filename']).reset_index(drop=True)
        
        # print(list(self.artist_frame['new_filename']))
        # print(len(self.artist_frame['new_filename']))
        if '81823.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '81823.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        if '95010.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '95010.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        if '50420.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '50420.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        if '98873.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '98873.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        if '82594.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '82594.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        if '33557.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '33557.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        if '72255.jpg' in list(self.artist_frame['new_filename']):
            idx = self.artist_frame[self.artist_frame['new_filename'] == '72255.jpg'].index
            self.artist_frame.drop(idx, inplace=True)
        # print(len(self.artist_frame['new_filename']))
        self.artist_frame = self.artist_frame.reset_index(drop=True)
        self.genre_dict = self.artist_frame['genre'].drop_duplicates().reset_index(drop='true').to_dict()
        # print('genre dict', self.genre_dict)


        self.root_dir = root_dir
        del to_remove
        del value_counts
        del style_threshold
        del genre_threshold
        self.train_transform = train_transform
        self.target_transform = target_transform

    def __len__(self):
        self.artist_frame = self.artist_frame.reset_index(drop=True)
        return len(self.artist_frame) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.artist_frame['new_filename'][idx])
        # ensure image stays or changes to PIL image or tensor
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # return None, None
            raise IOError(f'error reading image: {img_name}')

        genre = self.artist_frame['genre'][idx]
        genre_key = int(get_genre_key(self.genre_dict, genre))
        style = self.artist_frame['style'][idx]
        sample = {'image': image, 'genre_key': genre_key}

        if self.train_transform is not None:
            sample['image'] = self.train_transform(sample['image'])

        if self.target_transform is not None:
            sample['image'] = self.target_transform(sample['image'])

        return sample['image'], sample['genre_key']

def get_genre_key(genre_dict, val):
    for key, value in genre_dict.items():
         if val == value:
            return key
    return 1

def load_paintings_dataset(root_dir='paintings', train_transform=None, target_transform=None, train=True):
    csv_path = os.path.join(root_dir, 'all_data_info.csv')
    artist_frame = pd.read_csv(csv_path)
    # Filter out just the training data
    artist_train = artist_frame.loc[artist_frame['in_train']].reset_index(drop=True)
    # Remove the train and test, vs train only field, and the is training field.
    artist_train = artist_train.drop(['artist_group', 'in_train'], axis=1)
    # Keep only one entry per file
    artist_train = artist_train.drop_duplicates(subset=['new_filename']).reset_index(drop=True)
    del artist_frame
    artist_train.head()

    index_names = artist_train[artist_train['pixelsx'] >= 18000].index
    artist_train.drop(index_names, inplace = True)
    artist_train = artist_train.reset_index(drop=True)

    img_name = artist_train['new_filename'][0]
    genre = artist_train['genre'][0]
    style = artist_train['style'][0]

    # print('Image name: {}'.format(img_name))
    # print('Genre: {}'.format(genre))
    # print('Style: {}'.format(style))
    artist_train.describe()

    # Pick a threshold that makes other no more common than the least common
    genre_threshold = 1600


    value_counts = artist_train['genre'].value_counts() # Specific column
    to_remove = value_counts[value_counts <= genre_threshold].index
    artist_train['genre'].replace(to_remove, 'other', inplace=True)
    artist_train = artist_train.reset_index(drop='true')
    artist_train['genre'].value_counts().plot(kind='bar', title='genre')

    genre_dict = artist_train['genre'].drop_duplicates().reset_index(drop='true').to_dict()
    # Pick a threshold that makes other no more common than the most common
    style_threshold = 300

    value_counts = artist_train['style'].value_counts() # Specific column
    to_remove = value_counts[value_counts <= style_threshold].index
    artist_train['style'].replace(to_remove, 'other', inplace=True)
    del to_remove
    del value_counts
    artist_train['style'].value_counts().plot(kind='bar', title='style')

    del artist_train

    if train:
        paintings_dataset_train = PaintingsDataset(csv_file=csv_path, root_dir=os.path.join(root_dir, 'train'), train_transform=train_transform, target_transform=None, train=True)
        return paintings_dataset_train
    else:
        paintings_dataset_test = PaintingsDataset(csv_file=csv_path, root_dir=os.path.join(root_dir, 'test'), train_transform=None, target_transform=target_transform, train=False)
        return paintings_dataset_test

class NIHChestXrayDataset(Dataset):
    def __init__(
        self,
        data_dir,
        file_list,
        transform=None,
        class_names=None,
        minority_classes=None,
        remove_no_finding=True,
        use_clahe=False
    ):
        """
        If remove_no_finding=True, any image whose Finding Labels == "No Finding"
        will be removed from the dataset.
        If use_clahe=True, apply CLAHE to each image before transforms.
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.minority_classes = minority_classes if minority_classes else []
        self.remove_no_finding = remove_no_finding
        self.use_clahe = use_clahe

        csv_path = os.path.join(data_dir, "Data_Entry_2017_v2020.csv")
        meta_df = pd.read_csv(csv_path)

        # Filter meta_df by file_list
        file_set = set(file_list)
        meta_df = meta_df[meta_df["Image Index"].isin(file_set)].copy()
        meta_df.reset_index(drop=True, inplace=True)

        # Optionally remove "No Finding"
        if remove_no_finding:
            mask_no_finding = meta_df["Finding Labels"].str.contains("No Finding")
            meta_df = meta_df[~mask_no_finding].reset_index(drop=True)

        # Gather all .png paths
        png_paths = glob.glob(os.path.join(data_dir, 'images*', '*', '*.png'))
        self.map_filename2path = {os.path.basename(p): p for p in png_paths}

        print('length of findings:', len(meta_df["Finding Labels"]))

        # Build class names if not provided
        if class_names is None:
            # Derive from these samples
            all_labels = set()
            for row in meta_df["Finding Labels"]:
                for lab in row.split("|"):
                    all_labels.add(lab)
            self.class_names = sorted(list(all_labels))
        else:
            self.class_names = class_names[:]

        # If removing no_finding, also remove 'No Finding' from class_names
        if remove_no_finding and "No Finding" in self.class_names:
            self.class_names.remove("No Finding")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.samples = meta_df

        # Create a binary matrix for labels
        self.labels_matrix = np.zeros((len(self.samples), len(self.class_names)), dtype=np.float32)
        for i, row in self.samples.iterrows():
            labels = row["Finding Labels"].split("|")
            for lab in labels:
                if lab in self.class_to_idx:
                    self.labels_matrix[i, self.class_to_idx[lab]] = 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_name = row["Image Index"]
        labels = self.labels_matrix[idx]

        img_path = self.map_filename2path.get(img_name, None)
        if not img_path:
            raise FileNotFoundError(f"No path found for {img_name}")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Optional: Apply CLAHE
        if self.use_clahe:
            img = apply_clahe(img)

        target = torch.tensor(labels, dtype=torch.float)

        # Identify if sample belongs to any minority class
        is_minority = any(
            labels[i] == 1.0 for i, cls in enumerate(self.class_names) if cls in self.minority_classes
        )

        # Apply transforms
        if self.transform:
            if is_minority:
                img = self.minority_transform(img)
            else:
                img = self.transform(img)

        return img, target

    def minority_transform(self, img):
        # Define additional augmentations for minority classes
        additional_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        return additional_transforms(img)
    
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to an RGB image using OpenCV.
    Returns the enhanced RGB image.
    """
    # Convert RGB -> LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return img_clahe


def load_file_lists(data_dir):
    """
    Reads train_val_list.txt and test_list.txt in data_dir.
    Returns two lists of filenames (e.g. ["00000003_000.png", ...]).
    """
    with open(os.path.join(data_dir, 'train_val_list.txt'), 'r') as f:
        train_val_list = f.read().split()
    with open(os.path.join(data_dir, 'test_list.txt'), 'r') as f:
        test_list = f.read().split()
    return train_val_list, test_list

def load_chestxray(root_dir='/mnt/sdg/adv_datasets', train_transform=None, target_transform=None, train=True):
    
    DATA_DIR = root_dir
    
    train_val_list, test_list = load_file_lists(root_dir)
    random.shuffle(train_val_list)
    # val_count = int(0.2 * len(train_val_list))
    # val_list = train_val_list[:val_count]
    # train_list = train_val_list[val_count:]

    # print("Train images:", len(train_list))
    # print("Val images:  ", len(val_list))
    # print("Test images: ", len(test_list))
    
    
    #############################
    # split into train and test
    #############################
    # print("Train images:", len(train_val_list))
    # print("Test images: ", len(test_list))

    # 8.2 Define minority classes if desired
    class_counts = {
        'No Finding': 60361, # to be removed
        'Infiltration': 19894,
        'Effusion': 13317,
        'Atelectasis': 11559,
        'Nodule': 6331,
        'Mass': 5782,
        'Pneumothorax': 5302,
        'Consolidation': 4667,
        'Pleural_Thickening': 3385,
        'Cardiomegaly': 2776,
        'Emphysema': 2516,
        'Edema': 2303,
        'Fibrosis': 1686,
        'Pneumonia': 1431,
        'Hernia': 227
    }
    minority_threshold = 1000
    minority_classes = [cls for cls, count in class_counts.items() if count < minority_threshold]
    # print("Minority Classes:", minority_classes)
    
    # remove 'No Finding'
    dummy_train = NIHChestXrayDataset(
        root_dir, train_val_list,
        minority_classes=minority_classes,
        remove_no_finding=True,
        use_clahe=True  # apply CLAHE
    )
    all_classes = dummy_train.class_names
    num_classes = len(all_classes)
    # print("All classes (after removing No Finding):", num_classes, all_classes)
    
    train_transforms = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        T.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    train_ds = NIHChestXrayDataset(
        root_dir, train_val_list,
        transform=train_transforms,
        class_names=all_classes,
        minority_classes=minority_classes,
        remove_no_finding=True,
        use_clahe=True # <-- apply CLAHE to training set
    )
    test_ds = NIHChestXrayDataset(
        root_dir, test_list,
        transform=val_transforms,
        class_names=all_classes,
        minority_classes=minority_classes,
        remove_no_finding=True,
        use_clahe=True  # <-- apply CLAHE to test set
    )


    all_classes = train_ds.class_names
    num_classes = len(all_classes)
    print(f'NIH Chest X-ray: {num_classes} classes, {all_classes}')
    return train_ds, test_ds



def train_dataset_selection(dataset_name=None, train_transform=None, target_transform=None, root_dir=None):
    train_dataset, test_dataset = None, None
    DATASET_ROOT = os.path.join(root_dir, dataset_name)

    train_transform, target_transform = train_transform, target_transform
    
    assert dataset_name in ['comics', 'imagenet','car','cub','car','air','paintings','chestxray'], 'Please provide a dataset name'
    # print(f'[datasets/train] loading {dataset_name} dataset...')

    if dataset_name in ['comics', 'imagenet']:
        train_folder = 'train'
        test_folder = 'test' if dataset_name == 'comics' else 'val'
        train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, train_folder), train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, test_folder), target_transform)

        print(f'---{dataset_name}---')
        print("Train data set length:", len(train_dataset))
        print("Validation data set length:", len(test_dataset))

    elif dataset_name == 'car':
        # train_dataset, test_dataset = dataset.get_car(None, train=True, dataset=True), dataset.get_car(None, train=False, dataset=True)

        data = 'STCAR'
        dataset = 'STCAR'
        swap_num = [7,7]
        backbone = 'resnet50'
        # train dataset
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'train')
        
        anno = pd.read_csv(os.path.join(Config.anno_root, 'train.txt'),\
                                        sep=" ",\
                                        header=None,\
                                        names=['ImageName', 'label'])

        transformers = config.load_data_transformers(512, 448, [7,7])
        train_dataset = dataset_DCL.dataset(Config,
                        anno=anno,
                        swap=transformers["None"],
                        totensor=transformers['train_totensor'],
                        test=True)
        
        # test dataset
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                        sep=" ",\
                                        header=None,\
                                        names=['ImageName', 'label'])

        transformers = config.load_data_transformers(512, 448, [7,7])
        test_dataset = dataset_DCL.dataset(Config,
                        anno=anno,
                        swap=transformers["None"],
                        totensor=transformers['test_totensor'],
                        test=True)

        print('---Stanford Cars---')
        print("Train data set length:", len(train_dataset))
        print("Validation data set length:", len(test_dataset))

    elif dataset_name == 'cub':
        # train_dataset, test_dataset = dataset.get_cub(None, train=True, dataset=True), dataset.get_cub(None, train=False, dataset=True)

        data = 'CUB'
        dataset = 'CUB'
        swap_num = [7,7]
        backbone = 'resnet50'
        # train dataset
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'train')
        
        anno = pd.read_csv(os.path.join(Config.anno_root, 'train.txt'),\
                                        sep=" ",\
                                        header=None,\
                                        names=['ImageName', 'label'])

        transformers = config.load_data_transformers(512, 448, [7,7])
        train_dataset = dataset_DCL.dataset(Config,
                        anno=anno,
                        swap=transformers["None"],
                        totensor=transformers['train_totensor'],
                        test=True)
        
        # test dataset
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                        sep=" ",\
                                        header=None,\
                                        names=['ImageName', 'label'])

        transformers = config.load_data_transformers(512, 448, [7,7])
        test_dataset = dataset_DCL.dataset(Config,
                        anno=anno,
                        swap=transformers["None"],
                        totensor=transformers['test_totensor'],
                        test=True)

        print('---CUB-200-2011---')
        print("Train data set length:", len(train_dataset))
        print("Test data set length:", len(test_dataset))

    elif dataset_name == 'air':
        # train_dataset, test_dataset = dataset.get_air(None, train=True, dataset=True), dataset.get_air(None, train=False, dataset=True)

        data = 'AIR'
        dataset = 'AIR'
        swap_num = [7,7]
        backbone = 'resnet50'
        # train dataset
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'train')
        
        anno = pd.read_csv(os.path.join(Config.anno_root, 'train.txt'),\
                                        sep=" ",\
                                        header=None,\
                                        names=['ImageName', 'label'])

        transformers = config.load_data_transformers(512, 448, [7,7])
        train_dataset = dataset_DCL.dataset(Config,
                        anno=anno,
                        swap=transformers["None"],
                        totensor=transformers['train_totensor'],
                        test=True)
        
        # test dataset
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                        sep=" ",\
                                        header=None,\
                                        names=['ImageName', 'label'])

        transformers = config.load_data_transformers(512, 448, [7,7])
        test_dataset = dataset_DCL.dataset(Config,
                        anno=anno,
                        swap=transformers["None"],
                        totensor=transformers['test_totensor'],
                        test=True)
        
        print('---FGVC Aircraft---')
        print("Train data set length:", len(train_dataset))
        print("Test data set length:", len(test_dataset))

    elif dataset_name == 'paintings':
        train_dataset = load_paintings_dataset(root_dir=DATASET_ROOT, train_transform=train_transform, target_transform=None, train=True)
        test_dataset = load_paintings_dataset(root_dir=DATASET_ROOT, train_transform=None, target_transform=target_transform, train=False)

        print('---Paintings---')
        print("Train data set length:", len(train_dataset))
        print("Test data set length:", len(test_dataset))
    
    elif dataset_name == 'chestxray':
        # from ChestX import dataset
        DATASET_ROOT = DATASET_ROOT.replace('chestxray', 'CXR8')
        train_dataset, test_dataset = load_chestxray(root_dir=DATASET_ROOT, train_transform=train_transform, target_transform=target_transform, train=True)
        
        print('---ChestX---')
        print("Train data set length:", len(train_dataset))
        print("Test data set length:", len(test_dataset))
        pass
    
    else:
        raise NotImplementedError('Invalid dataset name!')
        exit(1)
        
    return train_dataset, test_dataset

if __name__ == '__main__':
    from paths import DATA_ROOT_DIR as DATA_ROOT
    train_dataset, test_dataset = train_dataset_selection(dataset_name='car', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='cub', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='air', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='paintings', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='comics', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='chestxray', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='imagenet', root_dir=DATA_ROOT)

    # python datasets/train.py
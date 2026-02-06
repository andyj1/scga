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

try:
    from datasets.DCL_finegrained import dataset, config, dataset_DCL
except:
    from DCL_finegrained import dataset, config, dataset_DCL 

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

def train_dataset_selection(dataset_name=None, train_transform=None, target_transform=None, root_dir=None):
    train_dataset, test_dataset = None, None
    DATASET_ROOT = os.path.join(root_dir, dataset_name)

    train_transform, target_transform = train_transform, target_transform
    
    assert dataset_name in ['comics', 'imagenet','car','cub','car','air','paintings','chestxray'], 'Please provide a dataset name'
    # print(f'[datasetstrain] loading {dataset_name} dataset...')

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
    
    else:
        raise NotImplementedError('Invalid dataset name!')
        exit(1)
        
    return train_dataset, test_dataset

if __name__ == '__main__':
    from paths import DATA_ROOT_DIR as DATA_ROOT
    train_dataset, test_dataset = train_dataset_selection(dataset_name='car', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='cub', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='air', root_dir=DATA_ROOT)
    train_dataset, test_dataset = train_dataset_selection(dataset_name='imagenet', root_dir=DATA_ROOT)

    # python datasets/train.py
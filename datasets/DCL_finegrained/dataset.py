from random import sample
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import os
import pandas as pd
from . import dataset_DCL
from . import config



def get_cub(batch_size, **kwargs):
    # args = parse_args()
    # args.data = 'CUB'
    # args.dataset = 'CUB'
    # Config = config.LoadConfig(args, 'test')
    data = 'CUB'
    dataset = 'CUB'
    swap_num = [7,7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                    sep=" ",\
                                    header=None,\
                                    names=['ImageName', 'label'])

    transformers = config.load_data_transformers(512, 448, [7,7])
    data_set = dataset_DCL.dataset(Config,
                       anno=anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                            #  collate_fn=dataset_DCL.collate_fn4test)
                                             collate_fn=dataset_DCL.collate_fn4test_path)

    return dataloader

def get_car(batch_size, **kwargs):
    # args = parse_args()
    # args.data = 'STCAR'
    # args.dataset = 'STCAR'
    # # args.data = data_name
    # Config = config.LoadConfig(args, 'test')
    data = 'STCAR'
    dataset = 'STCAR'
    swap_num = [7,7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    # anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
    #                                 sep=" ",\
    #                                 header=None,\
    #                                 names=['ImageName', 'label'])

    # ============================================================
    # pd.set_option('mode.chained_assignment', 'warn') # SettingWithCopyWarning
    pd.set_option('mode.chained_assignment',  None) # turn off warnings

    from glob import glob
    dataroot = Config.rawdata_root # root to STCAR
    anno_csv = pd.read_csv(os.path.join(dataroot, 'anno_test.csv'), sep=",", header=None, usecols=[0,5])
    anno = pd.DataFrame(columns=['ImageName', 'label'])
    anno['ImageName'] = anno_csv[0]
    anno['label'] = anno_csv[5]
    
    filepaths = []
    for i, img in enumerate(sorted(glob(os.path.join(dataroot,'cars_test/*.jpg')))):
        filepaths.append(img)
        anno.at[i, 'ImageName'] = img
        # anno['ImageName'].at[i, 0] = img # requires pandas~=1.2.5 # not work on ~=2.2.3
    # ============================================================
    
    transformers = config.load_data_transformers(512, 448, [7,7])
    data_set = dataset_DCL.dataset(Config,
                       anno=anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                            #  collate_fn=dataset_DCL.collate_fn4test)
                                             collate_fn=dataset_DCL.collate_fn4test_path)

    return dataloader



def get_air(batch_size, **kwargs):
    # args = parse_args()
    # args.data = 'AIR'
    # args.dataset = 'AIR'
    # Config = config.LoadConfig(args, 'test')
    data = 'AIR'
    dataset = 'AIR'
    swap_num = [2,2]
    backbone = 'resnet50'
    rawdata_root = 'None'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    anno = pd.read_csv(os.path.join(Config.anno_root, 'test.txt'),\
                                    sep=" ",\
                                    header=None,\
                                    names=['ImageName', 'label'])

    transformers = config.load_data_transformers(512, 448, [2,2])
    data_set = dataset_DCL.dataset(Config,
                       anno=anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor'],
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                            #  collate_fn=dataset_DCL.collate_fn4test)
                                             collate_fn=dataset_DCL.collate_fn4test_path)

    return dataloader


if __name__ == '__main__':
    data = get_cub(2)
    for img, label in data:
        print(img.shape)
        print(label.shape)
import torch
import os, sys
sys.path.append('..')
try:
    from datasets.paths import DATA_ROOT_DIR
except:
    from paths import DATA_ROOT_DIR

def eval_dataset_selection(name, batch_size, train=False, val=True):
    data_length = 0
    if name == 'imagenet':
        try:
            from datasets.imagenet import dataset
        except:
            from imagenet import dataset
        ds_fetcher = dataset.get
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val)
        # data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        data_length = 50000
        
    elif name == 'imagenet_incv3':
        try:
            from datasets.imagenet import dataset
        except:
            from imagenet import dataset
        ds_fetcher = dataset.get
        ds_val = ds_fetcher(batch_size=batch_size, input_size=299, train=train, val=val)
        # data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        data_length = 50000
        
    elif name == 'dcl_cub':
        try:
            from datasets.DCL_finegrained import dataset
        except:
            from DCL_finegrained import dataset
        ds_fetcher = dataset.get_cub
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val, rawdata_root=DATA_ROOT_DIR)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
    
    elif name == 'dcl_car':
        try:
            from datasets.DCL_finegrained import dataset
        except: 
            from DCL_finegrained import dataset
        ds_fetcher = dataset.get_car
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val, rawdata_root=DATA_ROOT_DIR)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        
    elif name == 'dcl_air':
        try:
            from datasets.DCL_finegrained import dataset
        except:
            from DCL_finegrained import dataset
        ds_fetcher = dataset.get_air
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val, rawdata_root=DATA_ROOT_DIR)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
            
    elif name == 'cifar10':
        try:
            from datasets.cifar import model, dataset
        except:
            from cifar import model, dataset
        ds_fetcher = dataset.get10 # prints 'building..'
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        
    elif name == 'cifar100':
        try:
            from datasets.cifar import model, dataset
        except:
            from cifar import model, dataset
        ds_fetcher = dataset.get100
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        
    elif name == 'svhn':
        try:
            from datasets.svhn import dataset
        except:
            from svhn import dataset
        ds_fetcher = dataset.get
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        
    elif name == 'stl10':
        try:
            from datasets.stl10 import dataset
        except:
            from stl10 import dataset
        ds_fetcher = dataset.get
        ds_val = ds_fetcher(batch_size=batch_size, train=train, val=val)
        data_length = len(ds_fetcher(batch_size=1, train=train, val=val))
        
    print("Validation {} data length: {}".format(name, data_length))
    return ds_val, data_length

if __name__ == '__main__':
    eval_dataset_selection('cifar10', 1)
    eval_dataset_selection('cifar100', 1)
    eval_dataset_selection('svhn', 1)
    eval_dataset_selection('stl10', 1)
    
    eval_dataset_selection('imagenet', 1)
    
    eval_dataset_selection('dcl_cub', 1)
    eval_dataset_selection('dcl_car', 1)
    eval_dataset_selection('dcl_air', 1)
    
        


import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from utee import misc
import torch
import os
from . import config
from . import LoadModel

from datasets.paths import DATA_ROOT_DIR
DCL_MODEL_PATH = f'{DATA_ROOT_DIR}/DCL_models'

def CUB(model):
    data = 'CUB'
    dataset = 'CUB'
    swap_num = [7,7]
    if model == 'resnet50':
        backbone = 'resnet50'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        Config.cls_2xmul = True
        model = LoadModel.MainModel(Config)
        model_dict=model.state_dict()
        pretrained_dict= torch.load(f'{DCL_MODEL_PATH}/CUB_Res_87.35.pth')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    elif model == 'senet154':
        backbone = 'senet154'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        # Config = config.LoadConfig(args, 'test')
        Config.cls_2xmul = True
        model2 = LoadModel.MainModel(Config)
        model2_dict=model2.state_dict()
        pretrained_dict2= torch.load(f'{DCL_MODEL_PATH}/CUB_SENet_86.81.pth')
        pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
        model2_dict.update(pretrained_dict2)
        model2.load_state_dict(model2_dict)
        model = model2

    elif model == 'se_resnet101':
        backbone = 'se_resnet101'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        # Config = config.LoadConfig(args, 'test')
        Config.cls_2xmul = True
        model3 = LoadModel.MainModel(Config)
        model3_dict=model3.state_dict()
        pretrained_dict3= torch.load(f'{DCL_MODEL_PATH}/CUB_SE_86.56.pth')
        pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
        model3_dict.update(pretrained_dict3)
        model3.load_state_dict(model3_dict)
        model = model3
    return model
    # return model, model2, model3

def CAR(model):
    data = 'STCAR'
    dataset = 'STCAR'
    swap_num = [7,7]
    
    if model == 'resnet50':
        backbone = 'resnet50'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')

        Config.cls_2xmul = True
        model = LoadModel.MainModel(Config)
        model_dict=model.state_dict()
        pretrained_dict= torch.load(f'{DCL_MODEL_PATH}/STCAR_Res_94.35.pth')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    elif model == 'senet154':
        backbone = 'senet154'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        Config.cls_2xmul = True
        model2 = LoadModel.MainModel(Config)
        model2_dict=model2.state_dict()
        pretrained_dict2= torch.load(f'{DCL_MODEL_PATH}/STCAR_SENet_93.36.pth')
        pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
        model2_dict.update(pretrained_dict2)
        model2.load_state_dict(model2_dict)
        model = model2
        
    elif model == 'se_resnet101':
        backbone = 'se_resnet101'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        Config.cls_2xmul = True
        model3 = LoadModel.MainModel(Config)
        model3_dict=model3.state_dict()
        pretrained_dict3= torch.load(f'{DCL_MODEL_PATH}/STCAR_SE_92.97.pth')
        pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
        model3_dict.update(pretrained_dict3)
        model3.load_state_dict(model3_dict)
        model = model3
        
    return model
    # return model, model2, model3


def AIR(model):
    data = 'AIR'
    dataset = 'AIR'
    swap_num = [2,2]
    if model == 'resnet50':
        backbone = 'resnet50'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        Config.cls_2xmul = True
        model = LoadModel.MainModel(Config)
        model_dict=model.state_dict()
        pretrained_dict= torch.load(f'{DCL_MODEL_PATH}/AIR_Res_92.23.pth')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    elif model == 'senet154':
        backbone = 'senet154'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        Config.cls_2xmul = True
        model2 = LoadModel.MainModel(Config)
        model2_dict=model2.state_dict()
        pretrained_dict2= torch.load(f'{DCL_MODEL_PATH}/AIR_SENet_92.08.pth')
        pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
        model2_dict.update(pretrained_dict2)
        model2.load_state_dict(model2_dict)
        model = model2
        
    elif model == 'se_resnet101':
        backbone = 'se_resnet101'
        Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
        Config.cls_2xmul = True
        model3 = LoadModel.MainModel(Config)
        model3_dict=model3.state_dict()
        pretrained_dict3= torch.load(f'{DCL_MODEL_PATH}/AIR_SE_91.90.pth')
        pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
        model3_dict.update(pretrained_dict3)
        model3.load_state_dict(model3_dict)
        model = model3

    return model
    # return model, model2, model3
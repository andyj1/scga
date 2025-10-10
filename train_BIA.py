import os
import argparse
import random
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from rich import print
import numpy as np
from PIL.Image import ImagePointHandler
from pytorch_lightning import seed_everything
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import datetime, logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import generator, surrogate
from myparser import get_parser

args = get_parser().parse_args()
seed_everything(args.seed)
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')


now = datetime.datetime.now()
now = now.strftime("%Y%m%d_T%H%M%S")[2:]
args.output_dir = '{}/{}_{}_{}_{}'.format(args.output_dir, now, args.method, args.generator, args.surrogate)
os.makedirs(args.output_dir, exist_ok=True, mode=0o777)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
log_file = os.path.join(args.output_dir, 'training.log')
file_handler = logging.FileHandler(log_file) # Create a file handler to log to output_dir
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()  # Create a console handler
console_handler.setLevel(logging.INFO)  # Set the logging level for the console
console_handler.setFormatter(formatter)  # Use the same formatter as the file handler
logger.addHandler(console_handler)  # Add the console handler to the logger
logger.info(args)


# Copy and save the current file to output_dir
import shutil
current_file_path = __file__  # Get the current file path
output_file_path = os.path.join(args.output_dir, os.path.basename(__file__))  # Define the output file path using the current filename
shutil.copy(current_file_path, output_file_path)  # Copy the current file to the output directory


if args.surrogate == 'vgg16':
    model = surrogate.Vgg16()
    layer_idx = 16  # Maxpooling.3
    # conv4-1 # 18 for LTP
elif args.surrogate == 'vgg19':
    model = surrogate.Vgg19()
    layer_idx = 18  # Maxpooling.3
elif args.surrogate == 'res152':
    model = surrogate.Resnet152()
    layer_idx = 5   # Conv3_8
elif args.surrogate == 'dense169':
    model = surrogate.Dense169()
    layer_idx = 6  # Denseblock.2
else:
    raise Exception('Please check the surrogate type')

model = model.cuda()
model.eval()

# Input dimensions
scale_size = 256
img_size = 224

# Generator
def create_generator(args):
    """Factory function to create the appropriate generator based on configuration."""
    
    if args.generator == 'resnet':
        netG = generator.GeneratorResnet().cuda()
        
    return netG
    # else:
    #     raise ValueError(f"Unknown generator type: {args.generator_type}")
netG = create_generator(args)
netG = torch.compile(netG)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Training Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

def normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t

# load train dataset
logger.info('loading training data...')
from datasets.paths import DATA_ROOT_DIR
from datasets.train import train_dataset_selection

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

train_set = train_dataset_selection(dataset_name=args.train_data, 
                                    root_dir=DATA_ROOT_DIR, 
                                    train_transform=data_transform)[0]
train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            num_workers=16, 
                                            pin_memory=True, 
                                            worker_init_fn=seed_worker,
                                            generator=g,)
train_size = len(train_set)
logger.info(f'Training data: {args.train_data}, data size: { train_size}')

# Loss
criterion = nn.CrossEntropyLoss()


args.output_dir = '{}/{}_{}_{}'.format(args.output_dir, args.method, args.generator, args.surrogate)
os.makedirs(args.output_dir, exist_ok=True)


# Training
loss_dict = {}
eps = args.eps/255.0
for epoch in range(args.epochs):
    cumulative_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', ncols=50)
    for i, (img, label) in enumerate(pbar):
        img = img.cuda()
        label = label.cuda()
        netG.train()
        optimG.zero_grad()
        adv, img_latent = netG(img, return_features=True)
        
        # unbounded
        adv_unbounded = adv.clone()

        # Projected (bounded): within eps and [0, 1]
        adv = torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)

        adv_out_slice = model(normalize(adv.clone()))[layer_idx]
        img_out_slice = model(normalize(img.clone()))[layer_idx]
        attention = torch.ones(adv_out_slice.shape).cuda() # filler for later integration with DA or RN module of BIA method

        loss_BIA = torch.cosine_similarity((adv_out_slice*attention).reshape(adv_out_slice.shape[0], -1), 
                                           (img_out_slice*attention).reshape(img_out_slice.shape[0], -1)).mean()
        
        loss = loss_BIA
        loss_dict['BIA'] = loss_BIA.item()
        
        loss.backward()
        optimG.step()

        cumulative_loss += abs(loss.item())
        
        if i > 0 and i % args.print_freq == 0:
            print_str = f'Epoch: {epoch+1}/{args.epochs}, Batch: {i+1}/{len(train_loader)}, Loss: {cumulative_loss / (i+1):.5f}'
            for key, value in loss_dict.items():
                print_str += f', {key}: {value:.5f}'
            logger.info(print_str)

        if i > 0 and i % args.save_freq == 0:
            torch.save(netG.state_dict(), 
                       os.path.join(args.output_dir, 
                                    'netG_{}_{}.pth'.format(args.method, epoch)))


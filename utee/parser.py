import argparse

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
            
    parser = argparse.ArgumentParser(description='SCGA')
    parser.add_argument('--layer_idx',
                        default=None,
                        type=int,
                        help='layer idx of surrogate model. if None, then automatically selects the default.')
    parser.add_argument('--tau', 
                        default=None, 
                        type=float)
    parser.add_argument('--train_data', 
                        default='imagenet', 
                        choices=['imagenet'], 
                        help='training data') 
    parser.add_argument('--method',
                        default=None,
                        help='method')
    parser.add_argument('-b', '--batch_size', 
                        type=int, 
                        default=16, 
                        help='Number of training samples/batch')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1, 
                        help='Number of training epochs')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0002, 
                        help='Initial learning rate for adam')
    parser.add_argument('--generator', 
                        type=str, 
                        default='resnet', 
                        help='generator model')
    parser.add_argument('--gap',
                        default=False,
                        action='store_true',
                        help='GAP generator model')
    parser.add_argument('--nat',
                        default=False,
                        action='store_true',
                        help='NAT generator model')
    parser.add_argument('--surrogate', 
                        type=str, 
                        default='vgg16',
                        help='Model against GAN is trained: vgg16, vgg19 res152, dense169')
    parser.add_argument('--RN', 
                        type=str2bool,
                        const=True,
                        default=False,
                        nargs="?",
                        help='If true, activating the Random Normalization module in training phase')
    parser.add_argument('--DA', 
                        type=str2bool,
                        const=True,
                        default=False,
                        nargs="?",
                        help='If true, activating the Domain-agnostic Attention module in training phase')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='GPU ID')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed')
    parser.add_argument('--save_freq',
                        type=int,
                        default=100,
                        help='save_freq')
    parser.add_argument('--print_freq',
                        type=int,
                        default=20,
                        help='print_freq')
    parser.add_argument('--vis_freq',
                        type=int,
                        default=200,
                        help='vis_freq')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./experiments',
                        help='experiment dir')
    parser.add_argument('--debug',
                        type=str2bool,
                        const=True,
                        default=False,
                        nargs="?",
                        help='debug mode')
    
    # vit generator params
    parser.add_argument('--image_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of patch')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input image channels')
    parser.add_argument('--embed_dim', type=int, default=768, help='The embedding dimension')
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer blocks')
    parser.add_argument('--heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=3072, help='Dimension of the MLP in the transformer block')
    parser.add_argument('--dim_head', type=int, default=64, help='Dimension of each attention head')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--eps', type=float, default=10, help='Maximum perturbation magnitude budget (L-infinity norm)')
    
    
    return parser
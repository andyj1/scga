import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib

from IPython import embed
from rich import print

def save_img_adv_diff(img, adv, save_path, sample_in_batch=0):
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    # Create a grid of images

    img_vis = img.permute(0, 2, 3, 1).detach().cpu().numpy()  # Convert to HWC format for visualization
    adv_vis = adv.permute(0, 2, 3, 1).detach().cpu().numpy()  # Convert to HWC format for visualization
    diff_vis = torch.abs(adv_vis - img_vis)  # Compute the difference
    diff_vis = (diff_vis - diff_vis.min()) / (diff_vis.max() - diff_vis.min())  # Normalize the difference
    
    img_grid = vutils.make_grid(img_vis, nrow=1, normalize=True, scale_each=True)
    adv_grid = vutils.make_grid(adv_vis, nrow=1, normalize=True, scale_each=True)
    diff_grid = vutils.make_grid(diff_vis, nrow=1, normalize=True, scale_each=True)

    # Plot the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_grid.permute(1, 2, 0).detach().cpu())
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_grid.permute(1, 2, 0).detach().cpu())
    plt.title('Adversarial Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff_grid.permute(1, 2, 0).detach().cpu())
    plt.title('Difference')
    plt.axis('off')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '{}.png'.format(i)), bbox_inches='tight')
    plt.close()
    
    
    # normalized_img = (img - img.min()) / (img.max() - img.min()) * 255
    # normalized_adv = (adv - adv.min()) / (adv.max() - adv.min()) * 255
    # img_vis = img[sample_in_batch, ...].permute(1,2,0).detach().cpu()
    # adv_vis = adv[sample_in_batch, ...].permute(1,2,0).detach().cpu()
    # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)  # Change RGB to BGR
    # adv_vis = cv2.cvtColor(adv_vis, cv2.COLOR_RGB2BGR)  # Change RGB to BGR
    # plt.subplot(131)
    # plt.imshow(img_vis)
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(adv_vis)
    # plt.axis('off')
    # plt.subplot(133)
    # normalized_diff = torch.abs(adv_vis - img_vis)
    # normalized_diff = (normalized_diff - normalized_diff.min()) / (normalized_diff.max() - normalized_diff.min()) * 255
    # normalized_diff = normalized_diff.byte()  # Convert to byte format for image representation
    # plt.imshow(normalized_diff)
    # plt.axis('off')
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, '{}.png'.format(i)), bbox_inches='tight')
    # plt.close()
    
    # from utee.misc import save_img_adv_diff
    # save_path = f'{args.output_dir}/sample_img'
    # os.makedirs(save_path, exist_ok=True)
    # save_img_adv_diff(img, adv, save_path, sample_in_batch=0)

def save_img(img, adv_unbounded, adv, save_dir, i):
    from torchvision.utils import save_image
    # img_output_dir = os.path.join(args.output_dir, 'wavelet_vis')
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    batch_size = img.shape[0]
    for j in range(img.shape[0]):
        if j == 0:
            batch_img_list = []
            batch_adv_unbounded_list = []
            # batch_adv_bounded_list = []
            batch_adv_list = []
        
        # Collect the j-th image in each list
        batch_img_list.append(img[j].permute(1,2,0))
        batch_adv_unbounded_list.append(adv_unbounded[j].permute(1,2,0))
        # batch_adv_bounded_list.append(adv_bounded[j].permute(1,2,0))
        batch_adv_list.append(adv[j].permute(1,2,0))

        # Once we're at the last item in the batch, save them all together
        if j == img.shape[0] - 1:
            batch_img_stack = torch.stack(batch_img_list, dim=0)                # [B, H, W, C]
            batch_adv_unbounded_stack = torch.stack(batch_adv_unbounded_list, dim=0)  # [B, H, W, C]
            # batch_adv_bounded_stack = torch.stack(batch_adv_bounded_list, dim=0)  # [B, H, W, C]
            batch_adv_stack = torch.stack(batch_adv_list, dim=0)                # [B, H, W, C]

            all_images = torch.cat([
                batch_img_stack,
                batch_adv_unbounded_stack,
                # batch_adv_bounded_stack,
                batch_adv_stack
            ], dim=0)                                       # [3B, H, W, C]
            all_images = all_images.permute(0, 3, 1, 2)    # [3B, C, H, W]

            save_image(
                all_images,
                os.path.join(save_dir, f'batch_{i}_all.png'),
                nrow=batch_size,
                normalize=True
            )
    return all_images
    
class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info
def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


def load_lmdb(lmdb_file, n_records=None):
    import lmdb
    import numpy as np
    lmdb_file = expand_user(lmdb_file)
    if os.path.exists(lmdb_file):
        data = []
        env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
        with env.begin() as txn:
            cursor = txn.cursor()
            begin_st = time.time()
            print("Loading lmdb file {} into memory".format(lmdb_file))
            for key, value in cursor:
                _, target, _ = key.decode('ascii').split(':')
                target = int(target)
                img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
                data.append((img, target))
                if n_records is not None and len(data) >= n_records:
                    break
        env.close()
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
        return data
    else:
        print("Not found lmdb file".format(lmdb_file))

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()

def eval_model(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    no_use = set(state_dict.keys()) - set(own_state.keys())
    if len(no_use) > 0:
        raise KeyError('some keys are not used: "{}"'.format(no_use))


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

# logging
import logging
import logging.config
from logging import StreamHandler, FileHandler, Handler, getLevelName
import sys
def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if save_dir:
        fh = FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info('Logging to %s', os.path.join(save_dir, filename))
    
    return logger

# Seed
import random
import torch
def setup_seed(seed=42):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    
import subprocess
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
CLOCK_SPEED = 1455 # Must choose a clock speed that's supported on your device.

def set_clock_speed():    
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling, but can help reduce variance.
    """
    # process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    # stdout, _ = process.communicate()
    process = subprocess.run(f"nvidia-smi -pm ENABLED -i {DEVICE}",      shell=True)
    process = subprocess.run(f"nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}", shell=True)

def reset_clock_speed():
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"nvidia-smi -rgc -i {DEVICE}", shell=True)
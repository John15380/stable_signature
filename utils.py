# utils.py (最终修复版 - 20251012)

import argparse
import json
import os
import random
import sys
import time
import datetime
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS

def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    """直接从根目录扫描图片文件，而不是从子目录。"""
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Exactly one of extensions and is_valid_file must be passed. HINT: extensions is a tuple e.g. ('.jpg', '.png')")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    
    cls_name = 'images'
    cls_index = class_to_idx.get(cls_name, 0)

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        if root != directory:
            continue
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = path, cls_index
                instances.append(item)
    return instances

class SimpleImageFolder(VisionDataset):
    """一个简化的ImageFolder，可以直接读取根目录下的图片，不需要子文件夹。"""
    def __init__(self, root, transform=None, target_transform=None,
                 loader=lambda path: Image.open(path).convert('RGB'),
                 is_valid_file=None, extensions=IMG_EXTENSIONS):
        super(SimpleImageFolder, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        classes = ['images']
        class_to_idx = {'images': 0}
        
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = f"Found 0 files in supplied directory: {self.root}\n"
            if extensions is not None:
                msg += f"Supported extensions are: {','.join(extensions)}"
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

def get_dataloader(path, transform, batch_size, num_imgs=None, shuffle=False, num_workers=0, collate_fn=None):
    # 使用我们自己的 SimpleImageFolder 替换官方的 ImageFolder
    dataset = SimpleImageFolder(path, transform=transform)

    # 只有当请求的图片数量小于数据集总数时，才进行随机抽样。
    if num_imgs is not None and num_imgs < len(dataset):
        print(f">>> Subsampling dataset from {len(dataset)} to {num_imgs} images...")
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return loader

def parse_params(param_str: str):
    """修复后的参数解析函数，能正确处理优化器名称。"""
    if not param_str:
        return {}
    
    parts = param_str.split(',')
    
    # 第一个部分如果没带'='，就认为是优化器名字
    if '=' not in parts[0]:
        name = parts[0]
        params_dict = {'name': name}
        parts = parts[1:]
    else: # 兼容老格式或者第一个就是带等号的参数
        params_dict = {}

    for p in parts:
        key, val = p.split('=')
        params_dict[key] = val
            
    return params_dict

def get_sha():
    import subprocess
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        uncommited_changes = subprocess.check_output(['git', 'status', '--porcelain'])
        status = 'has uncommited changes' if uncommited_changes else 'clean'
        return f'sha: {sha}, status: {status}, branch: {branch}'
    except Exception as e:
        return 'Not a git repo'

def bool_inst(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_optimizer(model_params, **optim_params):
    name = optim_params.pop('name')
    # 类型转换，特别是学习率
    for k, v in optim_params.items():
        try:
            optim_params[k] = float(v)
        except ValueError:
            pass
    return getattr(torch.optim, name)([p for p in model_params if p.requires_grad], **optim_params)

def adjust_learning_rate(optimizer, ii, steps, warmup_steps, base_lr):
    if ii < warmup_steps:
        lr = base_lr * (ii + 1) / warmup_steps
    else:
        # Cosine decay schedule
        progress = (ii - warmup_steps) / (steps - warmup_steps)
        lr = base_lr * 0.5 * (1. + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# From https://github.com/facebookresearch/deit/blob/main/utils.py
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque =_deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

from collections import deque as _deque

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = SmoothedValue(window_size=20, fmt='{value:.6f}')
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
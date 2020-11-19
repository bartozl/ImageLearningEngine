import requests
import os
import torch.nn as nn
import cv2
import numpy as np
import torch
from models import EmptyLayer, YoloLayer


URL_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
URL_WEIGHTS = "https://pjreddie.com/media/files/yolov3.weights"


def get_pretrained_weights(config, module_list):
    dest_path = f'{os.getcwd()}/yolov3.weights'
    if not os.path.exists(dest_path):
        print(f"downloading {URL_WEIGHTS}")
        r = requests.get(URL_CFG, allow_redirects=True)
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        print(f"saved in {dest_path}")

    with open(dest_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        pret_weights = np.fromfile(f, dtype=np.float32)

    idx = 0
    for o, (block, mod) in enumerate(zip(config, module_list)):
        if block['layer_type'] == 'convolutional':
            conv = mod[0]
            if 'batch_normalize' in block:
                bn = mod[1]

                bn_b = torch.from_numpy(pret_weights[idx:(idx + bn.bias.numel())]).view(bn.bias.shape)
                idx += bn.bias.numel()

                bn_w = torch.from_numpy(pret_weights[idx:(idx + bn.weight.numel())]).view(bn.weight.shape)
                idx += bn.weight.numel()

                bn_rm = torch.from_numpy(pret_weights[idx:(idx + bn.running_mean.numel())]).view(bn.running_mean.shape)
                idx += bn.running_mean.numel()

                bn_rv = torch.from_numpy(pret_weights[idx:(idx + bn.running_var.numel())]).view(bn.running_var.shape)
                idx += bn.running_var.numel()

                bn.bias.data.copy_(bn_b)
                bn.weight.data.copy_(bn_w)
                bn.running_mean.data.copy_(bn_rm)
                bn.running_var.data.copy_(bn_rv)
            else:
                bias = torch.from_numpy(pret_weights[idx:(idx + conv.bias.numel())]).view(conv.bias.shape)
                conv.bias.data.copy_(bias)
                idx += conv.bias.numel()
            weights = torch.from_numpy(pret_weights[idx:(idx + conv.weight.numel())]).view(conv.weight.shape)
            conv.weight.data.copy_(weights)
            idx += conv.weight.numel()


def download_cfg(dest_path):
    """
    Download the yolov3 configuration file
    :return: None
    """
    print(f"downloading {URL_CFG}")
    r = requests.get(URL_CFG, allow_redirects=True)
    with open(dest_path, 'wb') as f:
        f.write(r.content)
    print(f"saved in {dest_path}")


def rescale_anchors(anchors, stride):
    """
    The anchors are computed w.r.t. the input image (eg 416*416) then
    we need to rescale them in the current feature map size (e.g. 13 * 13)
    """
    num_anchors = len(anchors)
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    anchors = FloatTensor(np.asarray(anchors) / stride)
    anchors_w = anchors[..., 0].view(1, num_anchors, 1, 1)
    anchors_h = anchors[..., 1].view(1, num_anchors, 1, 1)
    return anchors_w, anchors_h


def create_grid(size_x):
    c_x = torch.ones((1, 1, size_x, size_x)) * torch.arange(size_x)
    c_y = (torch.ones((1, 1, size_x, size_x)) * torch.arange(size_x)).permute(0, 1, 3, 2)
    return c_x, c_y


def get_test_image(shape):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, shape).transpose((2, 0, 1))[np.newaxis, ...]  # H W C --> B C H W
    img = torch.from_numpy(img / 255.0).float()
    return img


def cast_type(attr, val):
    """
    :param attr: attribute of the block
    :param val: (str) value corresponding to that attribute
    :return: val correctly casted
    """
    # integer attributes
    if attr in ['batch', 'subdivisions', 'width', 'height', 'channels', 'angle', 'burn_in',
                'max_batches', 'batch_normalize', 'filters', 'size', 'stride', 'pad',
                'from', 'classes', 'num', 'truth_thresh', 'random']:
        val = int(val)
    # float attributes
    elif attr in ['momentum', 'decay', 'saturation', 'exposure',
                  'hue', 'learning_rate', 'jitter', 'ignore_thresh']:
        val = float(val)
    # string attributes
    elif attr in ['policy', 'activation']:
        val = str(val)
    # list of int attributes
    elif attr in ['layers', 'mask', 'steps']:
        val = [int(v) for v in val.split(',')] if len(val) >= 2 else int(val)
    # list of float
    elif attr in ['scales']:
        val = [float(v) for v in val.split(',')] if len(val) >= 2 else float(val)
    # list of list attributes
    elif attr == 'anchors':
        val = (val[1:].split(',  '))
        val2 = []  # TODO can I do it more pythonic? XD
        for x in val:
            val2.append([int(v) for v in x.split(',')])
        val = val2
    # an attribute has not been correctly parsed
    else:
        print('not classified', attr, val)
        exit(0)
    return attr, val


def parse_cfg(dest_path):
    """
    Parse the official configuration file 'yolov3.cfg'.
    :return: a list of dictionaries that represent each layer configuration
    """
    config = []
    if "yolov3.cfg" not in os.listdir(os.getcwd()):
        download_cfg(dest_path)
    with open(dest_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] == '[':  # possible layers: ['convolutional', 'route', 'yolo', shortcut', 'upsample']
                layer_type = lines[i].rstrip()[1:-1]  # extract layer type info
                block = {"layer_type": layer_type}  # new dictionary for each new block
                count = 1  # count the line for each block
                while i + count < len(lines) and lines[i + count][0] != '[':  # parse new line until the block ends
                    if lines[i + count][0] not in ['#', '\n']:  # skip commented or white lines
                        attr, val = lines[i + count].rstrip().split('=')
                        attr, val = cast_type(attr.rstrip(), val)
                        block[attr] = val
                    count += 1
                i += count
                config.append(block)

    return config


def create_modules(config, hyperparams):
    module_list = nn.ModuleList()
    prev_filters = hyperparams['channels']  # RGB images --> first layer has 3 filters
    output_filters = [prev_filters]  # store the output filters for each layer
    for idx, block in enumerate(config):
        mod = nn.Sequential()
        if block['layer_type'] == 'convolutional':
            filters = block['filters']
            pad = (block['size'] - 1) // 2 if block['pad'] else 0
            conv = nn.Conv2d(in_channels=prev_filters,
                             out_channels=filters,
                             kernel_size=block['size'],
                             stride=block['stride'],
                             padding=pad,
                             bias=not int(block.get('batch_normalize', 0)))
            mod.add_module(f"conv_{idx}", conv)

            if 'batch_normalize' in block:
                mod.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(filters))

            if block["activation"] == 'leaky':
                mod.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1, inplace=True))

        elif block['layer_type'] == 'upsample':
            mod.add_module(f'upsample_{idx}', nn.Upsample(scale_factor=block["stride"], mode='bilinear'))

        elif block['layer_type'] == 'route':
            filters: int = sum([output_filters[i] for i in list(block['layers'])])
            mod.add_module(f'route_{idx}', EmptyLayer())  # its just a placeholder.

        elif block['layer_type'] == 'shortcut':
            mod.add_module(f'shortcut_{idx}', EmptyLayer())

        elif block['layer_type'] == 'yolo':
            img_dim = (hyperparams['width'], hyperparams['height'])
            anchors = [block['anchors'][i] for i in block['mask']]
            mod.add_module(f'yolo_{idx}', YoloLayer(anchors, block['classes'], img_dim))
        else:
            print("Unknown layer type")
            exit(0)

        module_list.append(mod)
        prev_filters = filters
        output_filters.append(filters)

    return module_list

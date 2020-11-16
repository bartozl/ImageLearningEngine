import requests
import os
import torch.nn as nn
import cv2
import numpy as np
import torch


URL_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"


class EmptyLayer(nn.Module):
    """
    Dummy layer useful for route and shortcut layers
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x):
        pass


def get_test_image(shape):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, shape).transpose((2, 0, 1))[np.newaxis, ...]  # H W C --> B C H W
    img = torch.from_numpy(img/255.0).float()
    return img


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


def create_modules(config):
    module_list = nn.ModuleList()
    prev_filters = 3  # RGB images --> first layer has 3 filters
    output_filters = [prev_filters]  # store the output filters for each layer
    for idx, block in enumerate(config):
        mod = nn.Sequential()
        if block['layer_type'] == 'convolutional':
            filters = block['filters']
            pad = (block['size']-1)//2 if block['pad'] else 0
            conv = nn.Conv2d(in_channels=prev_filters,
                             out_channels=filters,
                             kernel_size=block['size'],
                             stride=block['stride'],
                             padding=pad,
                             bias=block.get('batch_normalize', 0))
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
            mod.add_module(f'yolo_{idx}', YoloLayer(block['anchors']))
        else:
            print("Unknown layer type")
            exit(0)

        module_list.append(mod)
        prev_filters = filters
        output_filters.append(filters)

    return module_list

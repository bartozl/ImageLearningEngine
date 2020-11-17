import requests
import os
import torch.nn as nn
import torch.nn.functional as F
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


def rescale_anchors(anchors, img_dim, grid_x):
    """
    The anchors are computed w.r.t. the input image (eg 416*416) then
    we need to rescale them in the current feature map size (e.g. 13 * 13)
    """
    num_anchors = len(anchors)
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    stride = img_dim[0] / grid_x  # how much the input image's dims have been reduced at this point
    anchors = FloatTensor(np.asarray(anchors) / stride)
    anchors_w = anchors[..., 0].view(1, num_anchors, 1, 1)
    anchors_h = anchors[..., 1].view(1, num_anchors, 1, 1)
    return anchors_w, anchors_h


def create_grid(size_x):
    c_x = torch.ones((1, 1, size_x, size_x)) * torch.arange(size_x)
    c_y = (torch.ones((1, 1, size_x, size_x)) * torch.arange(size_x)).permute(0, 1, 3, 2)
    return c_x, c_y


class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        """
        The anchors are the pre-defined bounding boxes, computed with k-means algorithm.
        """
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.img_dim = img_dim
        self.num_classes = num_classes

    def forward(self, x):
        """
        layer 82 --> x.shape = [batch, 255, 13, 13]
        layer 94 --> x.shape = [batch, 255, 26, 26]
        layer 106 --> x.shape = [batch, 255, 52, 52]

        The input channels are 255, in the form:
        C_in = bbox_num * (bbox_coords_pred + obj_score + classes_scores) = 3 * (4 + 1 + 80)
        where: bbox_coords_pred = (t_x, t_y, t_w, t_h) <-- offset w.r.t. the anchor boxes

        The output channels will be 255, in the form:
        C_out = bbox_num * (bbox_coords + obj_score + classes_scores) = 3 * (4 + 1 + 80)

        The C_out bbox_coords are computed as follow:
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * e^t_w
        b_h = p_h * e^t_h
        - c_x and c_y are the top left coordinate of the cells in the input grid
        - p_w and p_h are the scaled anchors width and height
        """
        batch_size = x.shape[0]
        num_anchors = len(self.anchors)
        num_classes = self.num_classes
        grid_x, grid_y = x.shape[2:]

        x = x.view(batch_size, num_anchors, num_classes + 5, grid_x, grid_y)  # x.shape = [batch, 3, 85, 13, 13]

        x = x.permute(0, 1, 3, 4, 2)  # x.shape = [batch, 3, 13, 13, 85]

        anchors_w, anchors_h = rescale_anchors(self.anchors, self.img_dim, grid_x)

        c_x, c_y = create_grid(grid_x)

        b_x = torch.sigmoid(x[..., 0]) + c_x  # shape = [1, 3, 13, 13]
        b_y = torch.sigmoid(x[..., 1]) + c_y  # shape = [1, 3, 13, 13]
        b_w = torch.exp(x[..., 2]) * anchors_w  # shape = [1, 3, 13, 13]
        b_h = torch.exp(x[..., 3]) * anchors_h  # shape = [1, 3, 13, 13]
        obj_score = x[..., 4]  # shape = [1, 3, 13, 13]
        class_score = x[..., 5:]  # shape = [1, 3, 13, 13, 80]

        output = torch.cat((b_x.unsqueeze(2),
                            b_y.unsqueeze(2),
                            b_w.unsqueeze(2),
                            b_h.unsqueeze(2),
                            obj_score.unsqueeze(2),
                            class_score.permute(0, 1, 4, 2, 3)),
                           dim=2).view(batch_size, num_anchors * (5 + num_classes), grid_x, grid_y)
        print(output.shape)

def get_test_image(shape):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, shape).transpose((2, 0, 1))[np.newaxis, ...]  # H W C --> B C H W
    img = torch.from_numpy(img / 255.0).float()
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

import requests
import os
import torch.nn as nn

URL_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"


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


def parse_cfg():
    """
    Parse the official configuration file 'yolov3.cfg'.
    :return: a list of dictionaries that represent each layer configuration
    """
    dest_path = os.getcwd() + "/yolov3.cfg"
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
                while i+count < len(lines) and lines[i+count][0] != '[':  # parse new line until the block ends
                    if lines[i+count][0] not in ['#', '\n']:  # skip commented or white lines
                        attr, val = lines[i + count].rstrip().split('=')
                        attr, val = cast_type(attr.rstrip(), val)
                        block[attr] = val
                    count += 1
                i += count
                config.append(block)

    return config


def create_modules(config):
    module_list = nn.ModuleList()
    for idx, block in enumerate(config[1:]):
        if block['layer_type'] == 'convolutional':
            pass
        if block['layer_type'] == 'upsample':
            pass
        if block['layer_type'] == 'route':
            pass
        if block['layer_type'] == 'shortcut':
            pass
        if block['layer_type'] == 'yolo':
            pass


config = parse_cfg()
create_modules(config)
print("DONE")

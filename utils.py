import requests
import os

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


def parse_cfg():
    """
    Parse the official configuration file in order to build the YOLO model.
    :return: a list of dictionaries that represent each layer configuration
    """
    # TODO return (config, hyperparams).  config: layers of the network,  hyperparams: general hyperparameters
    dest_path = os.getcwd() + "/yolov3.cfg"
    config = []
    if "yolov3.cfg" not in os.listdir(os.getcwd()):
        download_cfg(dest_path)
    with open(dest_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            layer_type = ''
            if lines[i][0] == '[':  # possible layers: ['convolutional', 'route', 'yolo', shortcut', 'upsample']
                layer_type = lines[i].rstrip()[1:-1]
                block = {"layer_type": layer_type}
                # TODO particular case "layer_type == 'net'" --> hyperparameters.
                count = 1
                while lines[i+count][0] not in ["\n", '', ' ', '#']:
                    # print(lines[i+count][0])
                    attr, val = lines[i + count].rstrip().split('=')
                    attr = attr.rstrip()
                    if attr in ['batch', 'subdivisions', 'width', 'height', 'channels', 'angle', 'burn_in',
                                'max_batches', 'batch_normalize', 'filters', 'size', 'stride', 'pad',
                                'from', 'classes', 'num', 'truth_thresh', 'random']:
                        val = int(val)
                    elif attr in ['momentum', 'decay', 'saturation', 'exposure',
                                  'hue', 'learning_rate', 'jitter', 'ignore_thresh']:
                        val = float(val)
                    elif attr in ['policy', 'activation']:
                        val = str(val)
                    elif attr in ['layers', 'mask']:
                        val = [int(v) for v in val.split(',')] if len(val) >= 2 else int(val)
                    elif attr == 'anchors':
                        val = (val[1:].split(',  '))
                        val2 = []  # TODO can I do it more pythonic? XD
                        for x in val:
                            val2.append([int(v) for v in x.split(',')])
                        val = val2
                    else:
                        print('not classified', attr, val)
                    block[attr] = val
                    count += 1
                i += count

                config.append(block)
        print(len(config))
        print(config[1])


parse_cfg()

print("DONE")

import utils
import os
from models import Darknet
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = f"{os.getcwd()}/yolov3.cfg"
config = utils.parse_cfg(config_path)
hyperparams = config.pop(0)

module_list = utils.create_modules(config, hyperparams)
weights = utils.get_pretrained_weights(config, module_list)

shape = (hyperparams['width'], hyperparams['height'])
img = utils.get_test_image(shape)  # easy test input

model = Darknet(config, module_list, device)
res = model(img)

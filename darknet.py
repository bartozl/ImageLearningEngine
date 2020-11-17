import utils
import torch
import torch.nn as nn
import os
import cv2


class Darknet(nn.Module):
    def __init__(self, config):
        super(Darknet, self).__init__()
        self.config = config
        self.hyperparams = self.config.pop(0)
        self.module_list = utils.create_modules(self.config, self.hyperparams)
        # print(self.module_list)

    def forward(self, x):
        config = self.config
        module_list = self.module_list
        layers_output = []
        for idx, mod in enumerate(module_list):
            layer_type = config[idx]['layer_type']
            if layer_type in ['convolutional', 'upsample']:
                if layer_type == 'yolo':
                    print(idx)
                x = mod(x)

            elif layer_type == 'route':
                # concatenate specific layers outputs in depth dimension
                x = torch.cat([layers_output[i] for i in config[idx]['layers']], dim=1)

            elif layer_type == 'shortcut':
                # sum specific layers outputs
                x = layers_output[-1] + layers_output[config[idx]['from']]

            elif layer_type == 'yolo':
                x = mod(x)

            layers_output.append(x)
        return x


config_path = f"{os.getcwd()}/yolov3.cfg"
config = utils.parse_cfg(config_path)
shape = (config[0]['width'], config[0]['height'])

img = utils.get_test_image(shape)  # easy test input

model = Darknet(config)
model(img)

import utils
import torch
import torch.nn as nn


class EmptyLayer(nn.Module):
    """
    Dummy layer useful for route and shortcut layers
    """

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        """
        The anchors are the pre-defined bounding boxes, computed with k-means algorithm.
        Instead of computing the bounding box from scratch, we adjust the dimension
        of this already know bounding boxes
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
        stride = self.img_dim[0] / grid_x  # how much the input image's dims have been reduced at this point

        x = x.view(batch_size, num_anchors, num_classes + 5, grid_x, grid_y)  # x.shape = [batch, 3, 85, 13, 13]

        x = x.permute(0, 1, 3, 4, 2)  # x.shape = [batch, 3, 13, 13, 85]
        anchors_w, anchors_h = utils.rescale_anchors(self.anchors, stride)

        c_x, c_y = utils.create_grid(grid_x)
        b_x = torch.sigmoid(x[..., 0]) + c_x  # shape = [1, 3, 13, 13]
        b_y = torch.sigmoid(x[..., 1]) + c_y
        b_w = torch.exp(x[..., 2]) * anchors_w  # shape = [1, 3, 13, 13]
        b_h = torch.exp(x[..., 3]) * anchors_h
        obj_score = torch.sigmoid(x[..., 4].contiguous())  # shape = [1, 3, 13, 13]
        class_score = torch.sigmoid(x[..., 5:].contiguous())  # shape = [1, 3, 13, 13, 80]

        b_x = b_x.view(batch_size, -1).unsqueeze(-1)  # shape = [1, 507, 1]
        b_y = b_y.view(batch_size, -1).unsqueeze(-1)
        b_w = b_w.view(batch_size, -1).unsqueeze(-1)
        b_h = b_h.view(batch_size, -1).unsqueeze(-1)
        obj_score = obj_score.view(batch_size, -1).unsqueeze(-1)
        class_score = class_score.view(batch_size, -1, num_classes)  # class_score.shape = [1, 507, 80]

        output = torch.cat([b_x, b_y, b_w, b_h, obj_score, class_score], dim=-1)  # [1, 507, 85]

        output[..., :4] *= stride

        return output


class Darknet(nn.Module):
    def __init__(self, config, module_list, device='cpu'):
        super(Darknet, self).__init__()
        self.config = config
        self.module_list = module_list
        self.device = device

    def forward(self, x):
        device = self.device
        config = self.config
        module_list = self.module_list
        layers_output, yolo_outputs = [], []

        for idx, mod in enumerate(module_list):
            layer_type = config[idx]['layer_type']
            if layer_type in ['convolutional', 'upsample']:
                x = mod(x.to(device))
            elif layer_type == 'route':
                # concatenate specific layers outputs in depth dimension
                x = torch.cat([layers_output[i] for i in config[idx]['layers']], dim=1).to(device)
            elif layer_type == 'shortcut':
                # sum specific layers outputs
                x = (layers_output[-1] + layers_output[config[idx]['from']]).to(device)

            elif layer_type == 'yolo':
                x = mod(x.to(device))
                yolo_outputs.append(x)
            layers_output.append(x)

        return torch.cat(yolo_outputs, dim=1)

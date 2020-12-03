from tensorflow import keras
from collections import defaultdict


class YoloLayer(keras.layers.Layer):
    def __init__(self, anchors, classes, img_dim):
        super(YoloLayer, self).__init__()
        """
        The anchors are the pre-defined bounding boxes, computed with k-means algorithm.
        Instead of computing the bounding box from scratch, we adjust the dimension
        of this already know bounding boxes
        """
        self.anchors = anchors
        self.classes = classes
        self.img_dim = img_dim

    @staticmethod
    def create_grid(batch, dim_x, dim_y, num_bbox=3):
        if batch is None:
            batch = 1
        grid_x = keras.backend.ones((batch, num_bbox, dim_x, dim_y)) * keras.backend.arange(float(dim_x))
        grid_y = keras.backend.ones((batch, num_bbox, dim_x, dim_y)) * keras.backend.arange(float(dim_y))
        grid_y = keras.layers.Permute((1, 3, 2))(grid_y)
        return grid_x, grid_y

    @staticmethod
    def scale_anchors(anchors, stride):
        """
        The anchors are computed w.r.t. the input image (eg 416*416) then
        we need to rescale them in the current feature map size (e.g. 13 * 13)
        """
        anchors_w = keras.backend.variable([a[0]//stride for a in anchors])
        anchors_w = keras.backend.reshape(anchors_w, shape=(1, len(anchors), 1, 1))

        anchors_h = keras.backend.variable([a[1]//stride for a in anchors])
        anchors_h = keras.backend.reshape(anchors_h, shape=(1, len(anchors), 1, 1))

        return anchors_w, anchors_h

    def call(self, x, training=False):
        """
        layer 82 --> x.shape = [batch, 255, 13, 13]
        layer 94 --> x.shape = [batch, 255, 26, 26]
        layer 106 --> x.shape = [batch, 255, 52, 52]

        The input channels are 255, in the form:
        C_in = bbox_num * (bbox_coords_pred + obj_score + classes_scores) = 3 * (4 + 1 + 80)
        where: bbox_coords_pred = (t_x, t_y, t_w, t_h) <--
        with: (t_x, t_y) offset w.r.t. to a fixed grid ('placed over the input')
              (t_w, t_h) offset w.r.t. the anchor boxes

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
        # x.shape = (None, 13, 13, 255)

        batch, dim_x, dim_y = x.shape[:3]  # dim_x = dim_y = 13
        stride = self.img_dim[0] // dim_x

        x = keras.layers.Reshape((len(self.anchors), dim_x, dim_y, 4 + 1 + self.classes))(x)

        anchors_w, anchors_h = self.scale_anchors(self.anchors, stride)
        c_x, c_y = self.create_grid(batch, dim_x, dim_y)

        b_x = keras.backend.sigmoid(x[..., 0]) + c_x  # [1, 3, 13, 13]
        b_y = keras.backend.sigmoid(x[..., 1]) + c_y
        b_w = keras.backend.exp(x[..., 2]) * anchors_w
        b_h = keras.backend.exp(x[..., 3]) * anchors_h
        obj_score = keras.backend.sigmoid(x[..., 4])
        class_score = keras.backend.sigmoid(x[..., 5:])  # [1, 3, 13, 13, 80]

        b_x = keras.backend.expand_dims(b_x, axis=-1)  # [1, 3, 13, 13, 1]
        b_y = keras.backend.expand_dims(b_y, axis=-1)
        b_w = keras.backend.expand_dims(b_w, axis=-1)
        b_h = keras.backend.expand_dims(b_h, axis=-1)
        obj_score = keras.backend.expand_dims(obj_score, axis=-1)

        x = keras.layers.Concatenate(axis=-1)([b_x * stride,
                                               b_y * stride,
                                               b_w * stride,
                                               b_h * stride,
                                               obj_score,
                                               class_score])

        x = keras.layers.Permute((2, 3, 1, 4))(x)
        x = keras.layers.Reshape(target_shape=(dim_x, dim_y, -1))(x)
        return x


def YoloNet(config, input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    layers_output, yolo_outputs = [], []
    idx = defaultdict(int)
    for block in config:
        if block['layer_type'] == 'convolutional':
            x = keras.layers.Conv2D(filters=block['filters'],
                                    kernel_size=block['size'],
                                    strides=block['stride'],
                                    padding='same' if (block['pad'] and block['stride']) else 'valid',
                                    use_bias=not int(block.get('batch_normalize', 0)),
                                    name=f'{idx["conv"]}_Conv2D'
                                    )(x)
            idx['conv'] += 1
            if 'batch_normalize' in block:
                x = keras.layers.BatchNormalization(name=f'{idx["bn"]}_bn')(x)
                idx['bn'] += 1
            if block['activation'] == 'leaky':
                x = keras.layers.LeakyReLU(alpha=0.1, name=f'{idx["leaky"]}_leaky')(x)
                idx['leaky'] += 1

        elif block['layer_type'] == 'upsample':
            s = block['stride']
            x = keras.layers.UpSampling2D(size=(s, s), name=str(idx['upsample']))(x)
            idx['upsample'] += 1

        elif block['layer_type'] == 'shortcut':
            layer_A, layer_B = layers_output[-1], layers_output[block['from']]
            x = keras.layers.Add(name=f'{idx["shortcut"]}_shortcut')([layer_A, layer_B])
            idx['shortcut'] += 1

        elif block['layer_type'] == 'route':
            if len(block['layers']) > 1:
                x = keras.layers.Concatenate(name=f'{idx["route"]}_route')([layers_output[i] for i in block['layers']])
            else:
                x = layers_output[block['layers'][0]]
            idx['route'] += 1

        elif block['layer_type'] == 'yolo':
            img_dim = input_shape[:2]
            anchors = [block['anchors'][i] for i in block['mask']]
            x = YoloLayer(anchors, block['classes'], img_dim)(x)
            x._name = str(idx['yolo'])
            idx['yolo'] += 1
            yolo_outputs.append(keras.layers.Reshape(target_shape=(-1, 85))(x))

        layers_output.append(x)

    model = keras.Model(inputs=inputs, outputs=yolo_outputs, name='YoloNet')
    return model


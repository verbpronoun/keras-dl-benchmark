import numpy as np

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import layers
from keras.layers.merge import concatenate, add
from keras.utils import np_utils
from keras import backend as K

if (K.image_data_format() == 'channels_first'):
    channel_axis = 1
else:
    channel_axis = 3

def my_conv(input, num_filters, kernel_size_tuple, strides=1, padding='valid'):
    x = Convolution2D(num_filters, kernel_size_tuple, strides=strides, padding=padding, 
                      use_bias=True, kernel_initializer='he_normal')(input)
    return x

def Block(input, numFilters, stride, isConvBlock, cardinality, bottleneck_width):
    expansion = 4

    width = (numFilters * bottleneck_width) / 64
    group = []

    for i in range(cardinality):
        # make grouped convolution
        x = my_conv(input, width, (1, 1), strides=stride)
        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=0.000011)(x)
        x = Activation('relu')(x)

        x = my_conv(x, width, (3, 3), padding='same')
        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=0.000011)(x)
        x = Activation('relu')(x)

        group.append(x)

    x = concatenate(group, axis=channel_axis)
    x = my_conv(x, expansion * numFilters, (1, 1))
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=0.000011)(x)

    if isConvBlock:
        shortcut = my_conv(input, expansion * numFilters, (1, 1), strides=stride)
        shortcut = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=0.000011)(shortcut)
    else:
        shortcut = input

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def make_layer(block, input, numFilters, numBlocks, stride, cardinality, bottleneck_width):
    x = block(input, numFilters, stride, True, cardinality, bottleneck_width)
    for i in range(numBlocks - 1):
        x = block(x, numFilters, 1, False, cardinality, bottleneck_width)
    return x

def ResNext_builder(block, num_blocks, input_shape, num_classes, cardinality, bottleneck_width):
    img_input = Input(shape=input_shape)
    x = my_conv(img_input, 64, (1, 1))
    x = BatchNormalization(axis = channel_axis, momentum=0.1, epsilon=0.000011)(x)
    x = Activation('relu')(x)
    
    x = make_layer(Block, x, 64, num_blocks[0], 1, cardinality, bottleneck_width)
    x = make_layer(Block, x, 128, num_blocks[1], 2, cardinality, bottleneck_width)
    x = make_layer(Block, x, 256, num_blocks[2], 2, cardinality, bottleneck_width)
    # x = make_layer(Block, x, 512, num_blocks[3], 2, cardinality, bottleneck_width)
    
    x = AveragePooling2D((8, 8), strides=8)(x)
    x = Flatten()(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=img_input, outputs=out)

def ResNext29(input_shape, num_classes, cardinality, bottleneck_width):
    return ResNext_builder(Block, (3, 3, 3), input_shape, num_classes, cardinality, bottleneck_width)

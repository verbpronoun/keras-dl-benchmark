import numpy as np

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K

if (K.image_data_format() == 'channels_first'):
    bn_axis = 1
else:
    bn_axis = 3


class BasicBlock():
    expansion = 1

    def __init__(self, input, numFilters, stride, isConvBlock):
        x = Convolution2D(numFilters, (3, 3), strides=stride, padding='same')(input)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)

        x = Convolution2D(numFilters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=bn_axis)(x)

        if isConvBlock:
            shortcut = Convolution2D(self.expansion * numFilters, (1, 1), strides = stride)(input)
            shortcut = BatchNormalization(axis=bn_axis)(x)
        else:
            shortcut = input

        x = layers.add([x, shortcut])
        self.out = Activation('relu')(x)

    def forward(self):
        return self.out


class PreActBlock():
    expansion = 1
 
    def __init__(self, input, numFilters, stride, isConvBlock):
        x = BatchNormalization(axis=bn_axis)(input)
        x = Activation('relu')(x)

        if isConvBlock:
            shortcut = Convolution2D(self.expansion * numFilters, (1, 1), strides = stride)(x)
        else:
            shortcut = x
        
        x = Convolution2D(numFilters, (3, 3), strides=stride, padding='same')(x)
 
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(numFilters, (3, 3), padding = 'same')(x)

        self.out = layers.add([x, shortcut])

    def forward(self):
        return self.out


class Bottleneck():
    expansion = 4
    def __init__(self, input, numFilters, stride, isConvBlock):
        x = Convolution2D(numFilters, (1, 1), strides=stride)(input)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)

        x = Convolution2D(numFilters, (3, 3), padding='same')(x) 
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)

        x = Convolution2D(4 * numFilters, (1, 1))(x)
        x = BatchNormalization(axis=bn_axis)(x)

        if isConvBlock:
            shortcut = Convolution2D(self.expansion * numFilters, (1, 1), strides=stride)(input)
            shortcut = BatchNormalization(axis=bn_axis)(shortcut)
        else:
            shortcut = input

        x = layers.add([x, shortcut])
        self.out = Activation('relu')(x)
    
    def forward(self):
        return self.out

class PreActBottleneck():
    expansion = 4

    def __init__(self, input, numFilters, stride, isConvBlock):
        x = BatchNormalization(axis=bn_axis)(input)
        x = Activation('relu')(x)

        if isConvBlock:
            shortcut = Convolution2D(self.expansion * numFilters, (1, 1), strides = stride)(x)
        else:
            shortcut = x

        x = Convolution2D(numFilters, (1, 1))(x)
 
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(numFilters, (3, 3), padding='same')(x) 

        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(4 * numFilters, (1, 1))(x)

    # did not finish...

class ResNet():
    def __init__(self, block, num_blocks, input_shape, num_classes):
        self.img_input = Input(shape=input_shape)
        x = Convolution2D(64, (3, 3), padding='same')(self.img_input)
        x = BatchNormalization(axis = bn_axis)(x)
        x = Activation('relu')(x)
        
        x = self.make_layer(block, x, 64, num_blocks[0], 1)
        x = self.make_layer(block, x, 128, num_blocks[1], 2)
        x = self.make_layer(block, x, 256, num_blocks[2], 2)
        x = self.make_layer(block, x, 512, num_blocks[3], 2)
        
        x = AveragePooling2D((4, 4), strides=2)(x)
        x = Flatten()(x)
        self.out = Dense(num_classes, activation='softmax')(x)
    
    def make_layer(self, block, input, numFilters, numBlocks, stride):
        x = block(input, numFilters, stride, True)
        for i in range(numBlocks - 1):
            x = block(x.forward(), numFilters, 1, False)
        return x.forward()
    
    def forward(self):
        return Model(inputs=self.img_input, outputs=self.out)
    
def ResNet18(input_shape, num_classes):
    resnet = ResNet(PreActBlock, [2, 2, 2, 2],input_shape, num_classes)
    return resnet.forward()

def ResNet34(input_shape, num_classes):
    resnet = ResNet(BasicBlock, [3, 4, 6 ,3],input_shape, num_classes)
    return resnet.forward()

def ResNet50(input_shape, num_classes):
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], input_shape, num_classes)
    return resnet.forward()
        
        

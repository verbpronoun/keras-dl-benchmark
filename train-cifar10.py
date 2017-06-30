import numpy as np

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10

batch_size = 32
num_epochs = 10
numClasses = 10
if (K.image_data_format() == 'channels_first'):
    bn_axis = 1
else:
    bn_axis = 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

numTrain, height, width, depth = x_train.shape
numTest = x_test.shape[0]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

y_train = np_utils.to_categorical(y_train, numClasses)
y_test = np_utils.to_categorical(y_test, numClasses)

img_input = Input(shape=(height, width, depth))


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

class ResNet():
    def __init__(self, block, num_blocks):
        x = Convolution2D(64, (3, 3), padding='same')(img_input)
        x = BatchNormalization(axis = bn_axis)(x)
        x = Activation('relu')(x)

        x = self.make_layer(block, x, 64, num_blocks[0], 1)
        x = self.make_layer(block, x, 128, num_blocks[1], 2)
        x = self.make_layer(block, x, 256, num_blocks[2], 2)
        x = self.make_layer(block, x, 512, num_blocks[3], 2)

        x = AveragePooling2D((4, 4), strides=2)(x)
        x = Flatten()(x)
        self.out = Dense(numClasses, activation='softmax')(x)

    def make_layer(self, block, input, numFilters, numBlocks, stride):
        x = block(input, numFilters, stride, True)
        for i in range(numBlocks - 1):
            x = block(x.forward(), numFilters, 1, False)
        return x.forward()

    def forward(self):
        return self.out

def ResNet18():
    resnet = ResNet(PreActBlock, [2, 2, 2, 2])
    return resnet.forward()

def ResNet34():
    resnet = ResNet(BasicBlock, [3, 4, 6 ,3])
    return resnet.forward()

def ResNet50():
    resnet = ResNet(Bottleneck, [3, 4, 6, 3])
    return resnet.forward()

model = Model(inputs=img_input, outputs = ResNet50())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
        

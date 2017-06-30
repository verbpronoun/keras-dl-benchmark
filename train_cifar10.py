import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import resnet_builder

batch_size = 32
num_epochs = 10
num_classes = 10
data_augmentation = True

# Preprocess train and test set data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

img_input = Input(shape=x_train.shape[1:])

model = resnet_builder.ResNet50(x_train.shape[1:], num_classes=num_classes)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not data_augmentation:
    history = model.fit(x_train, y_train, 
              batch_size=batch_size, 
              epochs=num_epochs, 
              verbose=1)
else:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, 
                                               batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=num_epochs)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
        

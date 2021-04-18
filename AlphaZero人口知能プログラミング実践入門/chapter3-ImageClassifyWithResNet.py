# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D, Add, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt



def run():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    #showImages(train_images, train_labels)
    shape = train_images.shape
    image_length = shape[1] * shape[2]
    
    # normalize
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # one-hot coding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    print('train images:', train_images.shape, 'labels:', train_labels.shape, 'test images:', test_images.shape, 'labels:', test_labels.shape)
    
    model = createModel(shape[1:])
    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['acc'])
    
    # learning
    history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split=0.1)

    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    # test
    loss, acc = model.evaluate(test_images, test_labels)
    print('loss:{:.3f}\nacc: {:.3f}'.format(loss, acc))
    
    # predict
    images = test_images[0:10]
    predictions = model.predict(images)
    predictions = np.argmax(predictions, axis=1)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predictions = [labels[n] for n in predictions]
    showImages(images, predictions)
    
def createModel(input_shape):
    
    def convolution(filters, kernel_size, strides=1):
        return Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))
    
    def blockA(filters, strides):
        def f(x):
            x = BatchNormalization()(x)
            b = Activation('relu')(x)
            x = convolution(filters // 4, 1, strides)(b)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            x = convolution(filters, 1, strides)(x)
            node = convolution(filters, 1, strides)(b)
            return Add()([x, node])
        return f
    
    def blockB(filters):
        def f(x):
            node = x
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = convolution(filters // 4, 1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = convolution(filters, 1)(x)

            return Add()([x, node])
        return f
    
    def residualBlock(filters, strides, unit_size):
        def f(x):
            x = blockA(filters, strides)(x)
            for i in range(unit_size - 1):
                x= blockB(filters)(x)
            return x
        return f
    
    
    inp = Input(shape=input_shape)
    x = convolution(16, 3)(inp)
    print(1, x.shape)    
    x = residualBlock(64, 1, 18)(x)
    print(2, x.shape)
    x = residualBlock(128, 1, 18)(x)
    print(3, x.shape)
    x = residualBlock(256, 1, 18)(x)
    
    print(4, x.shape)
    x = BatchNormalization(x)
    print(5, x.shape)
    x = Activation('relu')(x)
    print(6, x.shape)
    x = GlobalAveragePooling2D()(x)
    print(7, x.shape)
    outputs = Dense(10, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    model = Model(Inputs=inp, outputs=outputs)
    return model

    
    
def showImages(images, labels):
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title(str(labels[i]))
        plt.imshow(images[i], 'gray')
    plt.show()

# --------------------------------

    
if __name__ == '__main__':
    run()

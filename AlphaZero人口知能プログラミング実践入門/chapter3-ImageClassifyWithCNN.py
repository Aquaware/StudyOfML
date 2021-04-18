# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
    
    
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
    model = Sequential()
    
    print(input_shape)
    # Convolution
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    # Convolution
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    # MaxPool
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.25))
    
    # Convolution
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    # Convolution
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    # MaxPool
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.25))    
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
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

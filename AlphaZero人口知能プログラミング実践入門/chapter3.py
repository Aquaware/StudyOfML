# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt



def runMnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #showImages(train_images, train_labels)
    shape = train_images.shape
    image_size = shape[1] * shape[2]
    train_images = train_images.reshape((shape[0], shape[1] * shape[2]))
    train_labels = to_categorical(train_labels)
    shape = test_images.shape
    test_images = test_images.reshape((shape[0], shape[1] * shape[2]))
    test_labels = to_categorical(test_labels)
    params = {'input': [256, 'sigmoid', image_size],
             'hidden': [128, 'sigmoid'],
             'dropout':[0.5],
             'output': [10, 'softmax']}
    model = createModel(params)
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
    
    
    # learning
    history = model.fit(train_images, train_labels, batch_size=500, epochs=5, validation_split=0.2)
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
    images = [images[i].reshape((shape[1], shape[2])) for i in range(len(images))]
    showImages(images, predictions)
    
def createModel(params):
    model = Sequential()
    p = params['input']
    model.add(Dense(p[0], activation=p[1], input_shape=(p[2],)))
    p = params['hidden']
    model.add(Dense(p[0], activation=p[1]))
    p = params['dropout']
    model.add(Dropout(rate=p[0]))
    p = params['output']
    model.add(Dense(p[0], activation=p[1]))
    return model
    
    
def showImages(images, labels):
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title(str(labels[i]))
        plt.imshow(images[i], 'gray')
    plt.show()
    
if __name__ == '__main__':
    runMnist()

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D, Layer, Input, Conv2D, MaxPool2D, BatchNormalization, ReLU, Flatten, Dense, Add, Dropout
from tensorflow.keras.datasets import cifar10

class CNNModel(Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()                
        first_filters = 32
        conv_size = (3, 3)
        
        layers = [Conv2D(first_filters, input_shape=input_shape, kernel_size=conv_size, padding="same"),
                  BatchNormalization(),
                  Activation(tf.nn.relu),
                  MaxPool2D(pool_size=(2, 2), padding="same"),
                  Dropout(0.2)]
        
        for i in range(2):
            filters =  first_filters * (2**i)
            for _ in range(3):
                for _ in range(2):
                    layers.append(Conv2D(filters, kernel_size=conv_size))
                    layers.append(BatchNormalization())
                    layers.append(Activation(tf.nn.relu))
            layers.append(MaxPool2D(pool_size=(2, 2),  padding="same"))
            layers.append(Dropout(0.2))
        
        layers.append(Flatten())
        layers.append(Dropout(0.4))
        layers.append(Dense(100))
        layers.append(Activation(tf.nn.relu))
        layers.append(Dense(output_dim, activation=tf.nn.softmax))
        print('Units: ', len(layers))
        self._layers = layers
        
    def call(self, x):
        for layer in self._layers:
            if isinstance(layer, list):
                for l in layer:
                    x = l(x)    
            else:
                x = layer(x)
        return x
    
# -----

def loadDataSet():
    # データのロード
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

    # 0〜1へ正規化
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # one-hotエンコーディング
    Y_train = keras.utils.to_categorical(y_train)
    Y_test = keras.utils.to_categorical(y_test)

    input_shape = X_train.shape[1:]
    train_samples = X_train.shape[0]
    test_samples = X_test.shape[0]
    print('input shape:', input_shape, 'train samples:', train_samples, 'test samples:', test_samples)
    return (input_shape, (X_train, Y_train), (X_test, Y_test))

if __name__ == '__main__':
    # ハイパパラメータ
    epochs = 200
    batch_size = 50
    (input_shape, (X_train, Y_train), (X_test, Y_test)) = loadDataSet()
    model = CNNModel(input_shape, 10)
    shape= (None, input_shape[0], input_shape[1], input_shape[2])
    model.build(input_shape = shape)
    model.summary()
    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    #history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
    
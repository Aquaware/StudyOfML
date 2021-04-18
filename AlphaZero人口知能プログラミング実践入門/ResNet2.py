# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, ReLU, Flatten, Dense, Add, Dropout
from keras.datasets import cifar10

# データのロード
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
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

# ハイパパラメータ
epochs = 200
batch_size = 50

# モデルの生成
def generate_model(input_shape, block_f, blocks, block_sets, block_layers=2, first_filters=32, kernel_size=(3,3)):
  inputs = Input(shape=input_shape)
  
  # 入力層
  x = Conv2D(filters=first_filters, kernel_size=kernel_size, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPool2D((2, 2))(x)
  x = Dropout(0.2)(x)
  
  # 畳み込み層
  for s in range(block_sets):
    filters =  first_filters * (2**s)
    
    for b in range(blocks):
      x = block_f(x, kernel_size, filters, block_layers)
      
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)
  
  # 出力層
  x = Flatten()(x)
  x = Dropout(0.4)(x)
  x = Dense(100)(x)
  x = ReLU()(x)
  outputs = Dense(10, activation='softmax')(x)
  
  model = Model(input=inputs, output=outputs)
  
  return model

# shortcut connection無しのブロック
def plain_block(x, kernel_size, filters, layers):
  for l in range(layers):
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
  return x

# shortcut path有りのブロック (residual block)
def residual_block(x, kernel_size, filters, layers=2):
  shortcut_x = x
  
  for l in range(layers):
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    if l == layers-1:
      if K.int_shape(x) != K.int_shape(shortcut_x):
        shortcut_x = Conv2D(filters, (1, 1), padding='same')(shortcut_x)  # 1x1フィルタ
      
      x = Add()([x, shortcut_x])
      
    x = ReLU()(x)
    
  return x

# ハイパパラメータ
epochs = 200
batch_size = 50

# thinモデル
thin_model  = generate_model(input_shape, plain_block, blocks=1, block_sets=1)
thin_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
thin_history = thin_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
thin_model.summary()

# plainモデル
#plain_model  = generate_model(input_shape, plain_block, blocks=3, block_sets=2)
#plain_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#plain_history = plain_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
#print(plain_history)

# residualモデル
#residual_model  = generate_model(input_shape, residual_block, blocks=3, block_sets=2)
#residual_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#residual_history = residual_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))



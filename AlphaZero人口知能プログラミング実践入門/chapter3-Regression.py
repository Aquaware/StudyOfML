# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def run():
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    shape = train_data.shape
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(train_data, columns=columns)
    print(df.head())

    # shuffle
    size = train_labels.shape
    order = np.argsort(np.random.random(size))
    train_data =train_data[order]    
    train_labels = train_labels[order]
    
    # normalize
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    model = createModel()
    model.compile(loss='mse',  optimizer=Adam(lr=0.001), metrics=['mae'])
    stop = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, callbacks=[stop])
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_mae'], label='val mae')
    plt.xlabel('epoch')
    plt.ylabel('mae [100$]')
    
    # test
    test_loss, test_mae = model.evaluate(test_data, test_labels)
    print('loss:{:.3f}\nmae: {:.3f}'.format(test_loss, test_mae))
    
    # prediction
    prediction = model.predict(test_data[0:10]).flatten()
    print(np.round(test_labels[0:10]))
    print(np.round(prediction))
    
def createModel():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model
    
    

    
if __name__ == '__main__':
    run()

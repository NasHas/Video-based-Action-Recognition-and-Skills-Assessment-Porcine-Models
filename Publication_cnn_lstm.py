'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, TimeDistributed, Dropout, BatchNormalization, Bidirectional, \
    LSTM, CuDNNLSTM
from keras.regularizers import l2

def CARDIAC_cnn_lstm_net(Size, IMG_CHANNELS, n_classes):
 ##7.7 mio param. CNN + LSTM

    model = Sequential()
# In all lines the kernel_regularizer=l2(0.01), Dropout(0.2), BatchNormalization shoould be deleted, when using the network for action recognition.
# Also, when using the network for action recogntion, the input shape should be changed to 10, representing sequences of 10 seconds.
    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu',  kernel_regularizer=l2(0.01)), input_shape=(5, Size, Size, IMG_CHANNELS)))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dropout(0.4)))
    model.add(CuDNNLSTM(64))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(n_classes, activation='softmax'))

    return model



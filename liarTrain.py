import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from tokenization import normalize, process

# Bi LSTM, Scikit Learn, Keras
# Mixed data neural network
# https://medium.com/analytics-vidhya/adding-mixed-shaped-inputs-to-a-neural-network-5bafc58e9476


vocab_size = 20000  # use maxlen?
embedding_dim = 128
batch_size = 32

optimizer = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def sequenceModel(n_output):
    # add glove matrix?
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output, activation="softmax"))
    return model


# def clean_alt_list(list_):
#     list_ = list_.replace(', ', '","')
#     list_ = list_.replace('[', '["')
#     list_ = list_.replace(']', '"]')
#     return list_


# def to_1D(series):
#     return pd.Series([x for _list in series for x in _list])


def train1():
    print('Loading data...\n')
    liar_train = pd.read_csv(
        './cleanDatasets/tokenized_liar_train.csv')
    liar_train = shuffle(liar_train.reset_index(drop=True))
    x_train = np.array(liar_train['statement'])
    y_train = np.array(liar_train['label'])
    print(x_train.shape)
    print(y_train.shape)

    n_output = max(y_train)+1

    sequence_model = sequenceModel(n_output)
    sequence_model.summary()

    sequence_model.compile(loss=loss_function,
                           optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 1
    history = sequence_model.fit(
        x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)


def main():
    print('Loading data...\n')
    liar_train = normalize(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    x_train, y_train = process(liar_train)
    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)


main()

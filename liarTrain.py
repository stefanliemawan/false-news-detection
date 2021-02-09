import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from preprocess2 import returnStatementTokenizer, returnLabelEncoder
from ast import literal_eval

# Bi LSTM, Scikit Learn, Keras
# Mixed data neural network

vocab_size = 20000  # use maxlen?
embedding_dim = 128
batch_size = 32


def sequenceModel(n_output):
    # add glove matrix?
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(embedding_dim)))  # error
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output, activation="softmax"))
    return model


def main():
    print('Loading data...\n')
    liar_train = pd.read_pickle(
        './cleanDatasets/tokenized_liar_train.pkl')
    # liar_train = shuffle(liar_train.reset_index(drop=True))
    x_train = np.array(liar_train['statement'])
    y_train = np.array(liar_train['label'])
    print(x_train.shape)
    print(y_train.shape)

    n_output = max(y_train)+1

    sequence_model = sequenceModel(n_output)
    sequence_model.summary()

    sequence_model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])
    num_epochs = 10
    history = sequence_model.fit(
        x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)


main()

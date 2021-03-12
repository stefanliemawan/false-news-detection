import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from tensorflow.keras.constraints import max_norm, unit_norm


from tokenization import processPoliti, gloveMatrix, word2vecMatrix
from liarModel import readLiar
from train import train

# optimizer = tf.keras.optimizers.SGD()
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.RMSprop()
# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def politiModel(x1_shape, x2_shape, n_output1, n_output2, emb_matrix):

    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(emb_matrix.shape[0], emb_matrix.shape[1], weights=[
                       emb_matrix], trainable=True, name="embedding_1_1")(x1)
    cnn1_1 = Conv1D(128, 5,
                    activation="relu", padding="same", name="conv1d_1_1")(emb1_1)
    mp1_1 = MaxPooling1D(2, name="max_pooling1D_1_1")(cnn1_1)
    flat1_1 = Flatten(name="flat_1_1")(mp1_1)

    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(
        emb_matrix.shape[0], 5, trainable=True, name="embedding_2_1")(x2)
    cnn2_1 = Conv1D(64, 3, padding="same",
                    activation="relu", name="conv1d_2_1")(emb2_1)
    mp2_1 = MaxPooling1D(2, name="max_pooling1D_2_1")(cnn2_1)
    lstm_2_1 = LSTM(32, name="lstm_2_1")(mp2_1)

    x = concatenate([flat1_1, lstm_2_1])
    x = Dense(128, activation="relu")(x)

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def main():
    politi = pd.read_csv('./cleanDatasets/clean_politifact.csv')
    liar = readLiar()
    politi = politi.drop(['checker', 'date'], axis=1)
    liar = liar.drop(['id', 'context', 'subject',
                      "speaker's job title", 'state', 'party', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts'], axis=1)
    data = pd.concat([liar, politi], ignore_index=True)
    print("Full Data Shape = ", data.shape)
    print(data['label'].value_counts())
    num_epoch = 10
    train(data, processPoliti, politiModel, word2vecMatrix, num_epoch)
    # 0.65 val accuracy, 0.9 training accuracy
    # stop learning after around 10 epoch


main()

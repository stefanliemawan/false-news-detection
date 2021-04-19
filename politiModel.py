import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from tensorflow.keras.constraints import max_norm, unit_norm


from tokenization import processPoliti, gloveMatrix, word2vecMatrix, returnVocabSize
from train import train


# https://www.kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim
def politiModel(x1_shape, x2_shape, statement_vocab, metadata_vocab, n_output1, n_output2, emb_matrix):
    num_filters = 300
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(
        statement_vocab, emb_matrix.shape[1], weights=[emb_matrix], trainable=True, name="embedding_1_1")(x1)

    cnn1_1 = Conv1D(num_filters, 3,
                    activation="relu", name="conv1d_1_1")(emb1_1)
    mp1_1 = MaxPooling1D((x1_shape[1] - 3 + 1),
                         name="max_pooling1D_1_1")(cnn1_1)
    cnn1_2 = Conv1D(num_filters, 4,
                    activation="relu", name="conv1d_1_2")(emb1_1)
    mp1_2 = MaxPooling1D((x1_shape[1] - 4 + 1),
                         name="max_pooling1D_1_2")(cnn1_2)
    cnn1_3 = Conv1D(num_filters, 5,
                    activation="relu", name="conv1d_1_3")(emb1_1)
    mp1_3 = MaxPooling1D((x1_shape[1] - 5 + 1),
                         name="max_pooling1D_1_3")(cnn1_3)

    concat1_1 = concatenate([mp1_1, mp1_2, mp1_3])
    flat1_1 = Flatten(name="flat_1_1")(concat1_1)
    drop1_1 = Dropout(0.5, name="dropout_1_1")(flat1_1)

    # max x2 4500
    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(metadata_vocab, 100, trainable=True,
                       name="embedding_2_1")(x2)
    lstm2_1 = LSTM(256, name="lstm2_1")(emb2_1)
    drop2_1 = Dropout(0.5, name="dropout_2_1")(lstm2_1)

    x = concatenate([drop1_1, drop2_1])
    # x = concatenate([flat1_1, dense2_1])

    y1 = Dense(n_output1, activation="softmax", name="output_1")(x)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model

    # 3.6


# pd.set_option('display.max_rows', 10000)


def main():
    data = pd.read_csv('./cleanDatasets/clean_merged_politifact.csv')

    # data = data.head(5000)
    print(data.head())
    print("Full Data Shape = ", data.shape)
    # data.drop(data[data['label'] == 'FALSE'].iloc[3000:].index, inplace=True)
    # print("Full Data Shape = ", data.shape)
    print(data['label'].value_counts())
    print(data['subjectivity'].value_counts())

    num_epoch = 20
    train(data, processPoliti, politiModel, word2vecMatrix, num_epoch)


main()

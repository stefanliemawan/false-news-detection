import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import *


from tokenization import processLiar, gloveMatrix, word2vecMatrix
from train import train


def liarModelOld(x1_shape, x2_shape, n_output1, n_output2):
    # output2 static
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    cnn1_1 = Conv1D(128, 3,
                    activation="relu", name="conv1d_1_1")(x1)
    mp1_1 = MaxPooling1D(name="max_pooling1D_1_1")(cnn1_1)
    flat1_1 = Flatten(name="flat_1_1")(mp1_1)

    x2 = Input(shape=(x2_shape[1], 1), name="input_2")
    cnn2_1 = Conv1D(64, 3,
                    activation="relu", name="conv1d_2_1")(x2)
    bilstm2_1 = Bidirectional(LSTM(32, name="bi_lstm_2_1"))(cnn2_1)

    x = concatenate([flat1_1, bilstm2_1])
    x = Dense(128, activation="relu")(x)

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def liarModel(x1_shape, x2_shape, n_output1, n_output2, emb_matrix):
    # output2 static
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(emb_matrix.shape[0], emb_matrix.shape[1], weights=[
                       emb_matrix], trainable=False, name="embedding_1_1")(x1)
    cnn1_1 = Conv1D(128, 3,
                    activation="relu", padding="same", name="conv1d_1_1")(emb1_1)
    mp1_1 = MaxPooling1D(2, name="max_pooling1D_1_1")(cnn1_1)
    flat1_1 = Flatten(name="flat_1_1")(mp1_1)

    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(
        emb_matrix.shape[0], 10, trainable=True, name="embedding_2_1")(x2)
    cnn2_1 = Conv1D(32, 2, padding="same",
                    activation="relu", name="conv1d_2_1")(emb2_1)
    mp2_1 = MaxPooling1D(2, name="max_pooling1D_2_1")(cnn2_1)
    lstm_2_1 = LSTM(24, name="lstm_2_1")(mp2_1)

    x = concatenate([flat1_1, lstm_2_1])
    x = Dense(128, activation="relu")(x)

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def readLiar():
    liar_train = pd.read_csv(
        './cleanDatasets/clean_liar_train.csv')
    liar_test = pd.read_csv('./cleanDatasets/clean_liar_test.csv')
    liar_valid = pd.read_csv('./cleanDatasets/clean_liar_valid.csv')

    liar = pd.concat([liar_train, liar_test, liar_valid], ignore_index=True)
    return liar


# def main():
#     liar = readLiar()
#     train(liar, processLiar, liarModel, word2vecMatrix, 5)


# main()

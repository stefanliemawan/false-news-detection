import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from tensorflow.keras.constraints import max_norm, unit_norm


from tokenization import processPoliti, gloveMatrix, word2vecMatrix
from liarModel import readLiar
from train import train


# https://www.kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim
def politiModel(x1_shape, x2_shape, n_output1, n_output2, emb_matrix):
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(
        emb_matrix.shape[0], emb_matrix.shape[1], weights=[emb_matrix], trainable=True, name="embedding_1_1")(x1)

    cnn1_1 = Conv1D(128, 3,
                    activation="relu", name="conv1d_1_1")(emb1_1)
    mp1_1 = MaxPooling1D((x1_shape[1] - 3 + 1),
                         name="max_pooling1D_1_1")(cnn1_1)
    cnn1_2 = Conv1D(128, 4,
                    activation="relu", name="conv1d_1_2")(emb1_1)
    mp1_2 = MaxPooling1D((x1_shape[1] - 4 + 1),
                         name="max_pooling1D_1_2")(cnn1_2)
    cnn1_3 = Conv1D(128, 5,
                    activation="relu", name="conv1d_1_3")(emb1_1)
    mp1_3 = MaxPooling1D((x1_shape[1] - 5 + 1),
                         name="max_pooling1D_1_3")(cnn1_3)

    concat1_1 = concatenate([mp1_1, mp1_2, mp1_3], name="concat_1_1")
    flat1_1 = Flatten(name="flat_1_1")(concat1_1)
    drop1_1 = Dropout(0.5, name="dropout_1_1")(flat1_1)

    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(
        x2_shape[0], x2_shape[1], trainable=True, name="embedding_2_1")(x2)
    cnn2_1 = Conv1D(100, 3, activation="relu")(emb2_1)
    mp2_1 = MaxPooling1D(2, name="max_pooling1D_2_1")(cnn2_1)
    bilstm2_1 = Bidirectional(
        LSTM(64, recurrent_dropout=0.5), name="bilstm_2_1")(mp2_1)
    drop2_1 = Dropout(0.5, name="dropout_2_1")(bilstm2_1)

    x = concatenate([drop1_1, drop2_1])
    x = Dense(512, activation="relu")(x)

    y1 = Dense(n_output1, activation="softmax", name="output_1")(x)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model
    # how to increase accuracy and reduce overfitting

    # 50 epoch
    # x1 = 0.35 acc x2 = 0.66 - recurrent dropout 0.5


def main():
    data = pd.read_csv('./cleanDatasets/merged_politifact_tagged.csv')
    # data = pd.read_csv('./cleanDatasets/merged_politifact.csv')

    # data = data.head(5000)
    print("Full Data Shape = ", data.shape)
    print(data['label'].value_counts())
    print(data['subjectivity'].value_counts())
    num_epoch = 50
    train(data, processPoliti, politiModel, word2vecMatrix, num_epoch)
    # compare glove and word2vec


main()

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


def politiModel(x1_shape, x2_shape, n_output1, n_output2, emb_matrix):
    # OPTIMIZE, play with filter & kernel sizes, units, batch sizes, emb_matrix
    vocab_size = 19129
    d_scale = 0.25
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(
        100000, emb_matrix.shape[1], trainable=True, name="embedding_1_1")(x1)
    drop1_1 = Dropout(d_scale,
                      name="dropout_1_1")(emb1_1)
    cnn1_1 = Conv1D(128, 5,
                    activation="relu", padding="same", name="conv1d_1_1")(drop1_1)
    mp1_1 = MaxPooling1D(5, name="max_pooling1D_1_1")(cnn1_1)
    flat1_1 = Flatten(name="flat_1_1")(mp1_1)

    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(
        100000, 2, trainable=True, name="embedding_2_1")(x2)
    drop2_1 = Dropout(d_scale, name="dropout_2_1")(emb2_1)
    cnn2_1 = Conv1D(64, 3, padding="same",
                    activation="relu", name="conv1d_2_1")(drop2_1)
    mp2_1 = MaxPooling1D(2, name="max_pooling1D_2_1")(cnn2_1)
    drop2_2 = Dropout(d_scale, name="dropout_2_2")(mp2_1)
    lstm2_1 = LSTM(128, name="lstm_2_1")(drop2_1)

    x = concatenate([flat1_1, lstm2_1])
    x = Dense(512, activation="relu")(x)
    # Batch Normalization? Dropout doesn't really help
    # 100/100 [==============================] - 0s 3ms/step - loss: 2.4263 - output_1_loss: 1.3686 - output_2_loss: 1.0577 - output_1_accuracy: 0.6718 - output_2_accuracy: 0.8439
    # 90/90 [==============================] - 0s 3ms/step - loss: 2.4301 - output_1_loss: 1.3730 - output_2_loss: 1.0571 - output_1_accuracy: 0.6664 - output_2_accuracy: 0.8430

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def main():
    data = pd.read_csv('./cleanDatasets/merged_politifact.csv')
    print(len(data['statement']) -
          len(data['statement'].drop_duplicates()), "Duplicates column")
    data = data.drop_duplicates(subset=["statement"])
    print("Full Data Shape = ", data.shape)
    print(data['label'].value_counts())
    num_epoch = 21
    train(data, processPoliti, politiModel, word2vecMatrix, num_epoch)
    # 0.65 val accuracy, 0.9 training accuracy
    # 0.69 val accuracy, 0.8 training accuracy with 10
    # stop learning after around 10 epoch


main()

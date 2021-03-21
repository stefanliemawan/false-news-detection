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
    # 0.33 on 10 epoch
    # OPTIMIZE, play with filter & kernel sizes, units, batch sizes, emb_matrix
    vocab_size = 19129
    d_scale = 0.5
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(
        30000, emb_matrix.shape[1], trainable=True, name="embedding_1_1")(x1)
    drop1_1 = Dropout(d_scale,
                      name="dropout_1_1")(emb1_1)
    cnn1_1 = Conv1D(128, 3,
                    activation="relu", padding="same", name="conv1d_1_1")(drop1_1)
    mp1_1 = MaxPooling1D(2, name="max_pooling1D_1_1")(cnn1_1)
    bilstm1_1 = Bidirectional(LSTM(128, name="bilstm_1_1"))(mp1_1)
    bn1_1 = BatchNormalization(name="batch_normalization_1_1")(bilstm1_1)

    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(
        30000, 2, trainable=True, name="embedding_2_1")(x2)
    drop2_1 = Dropout(d_scale, name="dropout_2_1")(emb2_1)
    lstm2_2 = LSTM(128, name="lstm_2_2")(drop2_1)
    bn2_2 = BatchNormalization(name="batch_normalization_2_2")(lstm2_2)

    x = concatenate([bn1_1, bn2_2])
    x = BatchNormalization(name="batch_normalization")(x)
    x = Dense(128, activation="relu")(x)

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def main():
    data = pd.read_csv('./cleanDatasets/merged_politifact_no_duplicates.csv')
    # mask = (data.label == "FALSE")
    # idx, = np.where(mask)
    # data = data.drop(data.index[idx[:1400]])

    print("Full Data Shape = ", data.shape)
    print(data['label'].value_counts())
    num_epoch = 20
    train(data, processPoliti, politiModel, word2vecMatrix, num_epoch)

    # 30k data
    # 0.65 val accuracy, 0.9 training accuracy
    # 0.69 val accuracy, 0.8 training accuracy with 10

    # 20k data
    # 0.3 val accuracy, 0.9 training accuracy
    # 0.75 val accuracy, 0.89 training accuracy


main()

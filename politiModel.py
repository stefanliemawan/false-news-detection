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


from tokenization import processPoliti, processLiar, gloveMatrix, word2vecMatrix, fasttextMatrix, returnVocabSize
from train import train, trainLiar, trainPoliti


# https://www.kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim
def politiModel(x1_shape, x2_shape, x3_shape, statement_vocab, metadata_vocab, n_output1, n_output2, emb_matrix):
    cnn_filter = 100
    x1 = Input(shape=(x1_shape[1], ), name="input_1")
    emb1_1 = Embedding(
        statement_vocab, emb_matrix.shape[1], weights=[emb_matrix], trainable=True, name="embedding_1_1")(x1)

    cnn1_1 = Conv1D(cnn_filter, 3,
                    activation="relu", name="conv1d_1_1")(emb1_1)
    mp1_1 = MaxPooling1D((x1_shape[1] - 3 + 1),
                         name="max_pooling1D_1_1")(cnn1_1)
    cnn1_2 = Conv1D(cnn_filter, 4,
                    activation="relu", name="conv1d_1_2")(emb1_1)
    mp1_2 = MaxPooling1D((x1_shape[1] - 4 + 1),
                         name="max_pooling1D_1_2")(cnn1_2)
    cnn1_3 = Conv1D(cnn_filter, 5,
                    activation="relu", name="conv1d_1_3")(emb1_1)
    mp1_3 = MaxPooling1D((x1_shape[1] - 5 + 1),
                         name="max_pooling1D_1_3")(cnn1_3)

    concat1_1 = concatenate([mp1_1, mp1_2, mp1_3])
    flat1_1 = Flatten(name="flat_1_1")(concat1_1)
    # drop1_1 = Dropout(0.5, name="dropout_1_1")(flat1_1)

    # max x2 4500
    x2 = Input(shape=(x2_shape[1], ), name="input_2")
    emb2_1 = Embedding(metadata_vocab, 100, trainable=True,
                       name="embedding_2_1")(x2)
    lstm2_1 = LSTM(256, name="lstm2_1")(emb2_1)  # 256
    # drop2_1 = Dropout(0.5, name="dropout_2_1")(lstm2_1)

    x3 = Input(shape=(x3_shape[1],), name="input_3")
    # emb3_1 = Embedding(metadata_vocab, 100, trainable=True,
    #                    name="embedding_3_1")(x2)
    # bilstm3_1 = Bidirectional(LSTM(64, name="bilstm3_1"))(emb3_1)  # 256
    # drop3_1 = Dropout(0.5, name="dropout_3_1")(bilstm3_1)
    dense3_1 = Dense(256, activation="relu")(x3)

    # x = concatenate([drop1_1, drop2_1, dense3_1])
    x = concatenate([flat1_1, lstm2_1, dense3_1])
    x = Dense(256, activation="relu")(x)
    # no dropout 0.48 0.49

    y1 = Dense(n_output1, activation="softmax", name="output_1")(x)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(x)
    model = keras.Model(inputs=[x1, x2, x3], outputs=[y1, y2])
    return model

    # 3.6


# pd.set_option('display.max_rows', 10000)
def liar():
    liar_train = pd.read_csv(
        "./cleanDatasets/clean_liar_train.csv").reset_index(drop=True)
    liar_test = pd.read_csv(
        "./cleanDatasets/clean_liar_test.csv").reset_index(drop=True)
    liar_val = pd.read_csv(
        "./cleanDatasets/clean_liar_valid.csv").reset_index(drop=True)

    print(liar_train.isna().sum(), "\n")
    print(liar_test.isna().sum(), "\n")
    print(liar_val.isna().sum(), "\n")

    # nan = pd.DataFrame(columns=["columns", "train", "test", "val"])
    # nan["columns"] = liar_train.columns.values
    # nan["train"] = liar_train.isna().sum().tolist()
    # nan["test"] = liar_test.isna().sum().tolist()
    # nan["val"] = liar_val.isna().sum().tolist()

    # nan.to_csv("./datasets/LIAR/nan.csv", index=False)

    # print(nan)

    liar_train = liar_train[liar_train["false counts"].notna()]
    liar_test = liar_test[liar_test["false counts"].notna()]

    liar_train.fillna("None", inplace=True)
    liar_test.fillna("None", inplace=True)
    liar_val.fillna("None", inplace=True)

    print(liar_train['label'].value_counts())
    print(liar_train['subjectivity'].value_counts())

    num_epoch = 10
    trainLiar(liar_train, liar_test, liar_val, processLiar,
              politiModel, word2vecMatrix, num_epoch)


def politi():
    data = pd.read_csv(
        './cleanDatasets/clean_merged_politifact+_editeds.csv').reset_index(drop=True)
    # data = pd.read_csv(
    #     './cleanDatasets/clean_merged_politifact+.csv').reset_index(drop=True)

    # data = data.head(5000)
    # print(data.isna().sum())
    data["true counts"].fillna(0, inplace=True)
    data.fillna("None", inplace=True)

    print("Full Data Shape = ", data.shape)
    # data.drop(data[data['label'] == 'FALSE'].iloc[3000:].index, inplace=True)
    # print("Full Data Shape = ", data.shape)
    print(data['label'].value_counts())
    print(data['subjectivity'].value_counts())

    num_epoch = 21
    trainPoliti(data, processPoliti, politiModel, word2vecMatrix, num_epoch)


def main():
    # liar()
    politi()


main()

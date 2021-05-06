from train import train, trainLiar, trainPoliti
from tokenization import processPoliti, processLiar, gloveMatrix, word2vecMatrix, fasttextMatrix, returnVocabSize, returnBertLayer
from tensorflow.keras.constraints import max_norm, unit_norm
from sklearn.utils import shuffle
from keras.utils import np_utils
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub


def handleNaN(data):
    for header in data.columns.values:
        data[header] = data[header].fillna(
            data[header][data[header].first_valid_index()])
    return data


def politiModel(seq_length, x2_shape, x3_shape, x4_shape, metadata_vocab, n_output1, n_output2):

    # regularizer = regularizers.l2(1)
    regularizer = None

    input_word_ids = tf.keras.Input(
        shape=(seq_length, ), dtype=tf.int32)
    input_mask = tf.keras.Input(
        shape=(seq_length, ), dtype=tf.int32)
    input_type_ids = tf.keras.Input(
        shape=(seq_length, ), dtype=tf.int32)

    bert_layer = returnBertLayer()

    outputs = bert_layer(
        {"input_word_ids": input_word_ids, "input_mask": input_mask, "input_type_ids": input_type_ids})
    print(outputs)
    pooled_output = outputs["pooled_output"]
    dense1_1 = Dense(128, activation="relu")(pooled_output)
    dense1_2 = Dense(64, activation="relu")(dense1_1)
    # seq_output = outputs["sequence_output"]
    # lstm1_1 = LSTM(64, kernel_regularizer=regularizer)(seq_output)
    # clf_output = sequence_output[:, 0, :]

    x2 = Input(shape=(x2_shape[1], ))
    x3 = Input(shape=(x3_shape[1], ))
    x4 = Input(shape=(x4_shape[1], ))

    # emb = Embedding(
    #     statement_vocab, emb_matrix.shape[1], weights=[emb_matrix], trainable=True, name="embedding_1_1")

    # emb1_1 = emb(x1)

    # cnn1_1 = Conv1D(32, 3, kernel_regularizer=regularizer,
    #                 activation="relu", name="conv1d_1_1")(seq_output)
    # mp1_1 = MaxPooling1D((seq_length - 3 + 1),
    #                      name="max_pooling1D_1_1")(cnn1_1)
    # cnn1_2 = Conv1D(32, 4, kernel_regularizer=regularizer,
    #                 activation="relu", name="conv1d_1_2")(seq_output)
    # mp1_2 = MaxPooling1D((seq_length - 4 + 1),
    #                      name="max_pooling1D_1_2")(cnn1_2)
    # cnn1_3 = Conv1D(32, 5, kernel_regularizer=regularizer,
    #                 activation="relu", name="conv1d_1_3")(seq_output)
    # mp1_3 = MaxPooling1D((seq_length - 5 + 1),
    #                      name="max_pooling1D_1_3")(cnn1_3)

    # concat1_1 = concatenate([mp1_1, mp1_2, mp1_3], name="concat_1_1")
    # flat1_1 = Flatten(name="flatten_1_1")(concat1_1)

    # drop1_1 = Dropout(0.2, name="dropout_1_1")(flat1_1)

    # emb2_1 = Embedding(metadata_vocab, 100, trainable=True,
    #                    name="embedding_2_1")(x2)
    # lstm2_1 = LSTM(64,  kernel_regularizer=regularizer,
    #                name="lstm_2_1")(emb2_1)

    # dense3_1 = Dense(64, activation="relu", name="dense_3_1")(x3)

    # dense4_1 = Dense(64, activation="relu", name="dense_4_1")(x4)

    # x = concatenate([lstm1_1, lstm2_1, dense3_1])
    # x = Dense(x.shape[1], activation="relu")(x)

    y1 = Dense(n_output1, activation="softmax", name="output_1")(dense1_2)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(dense1_2)

    model = keras.Model(
        inputs={"input_word_ids": input_word_ids, "input_mask": input_mask, "input_type_ids": input_type_ids, "x2": x2, "x3": x3, "x4": x4}, outputs=[y1, y2])
    return model


# pd.set_option("display.max_rows", 10000)
def liar():
    liar_train = pd.read_csv(
        "./cleanDatasets/clean_liar_train.csv").reset_index(drop=True)
    liar_test = pd.read_csv(
        "./cleanDatasets/clean_liar_test.csv").reset_index(drop=True)
    liar_val = pd.read_csv(
        "./cleanDatasets/clean_liar_valid.csv").reset_index(drop=True)

    # liar_train = handleNaN(liar_train)

    # print(liar_train.isna().sum(), "\n")
    # print(liar_test.isna().sum(), "\n")
    # print(liar_val.isna().sum(), "\n")
    # print(nan)

    liar_train = liar_train[liar_train["false counts"].notna()]
    liar_test = liar_test[liar_test["false counts"].notna()]

    # liar_train["true counts"].fillna(0, inplace=True)
    # liar_test["true counts"].fillna(0, inplace=True)
    # liar_val["true counts"].fillna(0, inplace=True)

    liar_train.fillna("None", inplace=True)
    liar_test.fillna("None", inplace=True)
    liar_val.fillna("None", inplace=True)

    print(liar_train["label"].value_counts())
    print(liar_train["subjectivity"].value_counts())

    num_epoch = 100
    # best at 10 epoch
    trainLiar(liar_train, liar_test, liar_val, processLiar,
              politiModel, word2vecMatrix, num_epoch)


def politi():
    # data = pd.read_csv(
    #     "./cleanDatasets/clean_merged_politifact+_editeds_shuffled.csv").reset_index(drop=True)
    data = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact+.csv").reset_index(drop=True)

    # data = handleNaN(data)

    # data = data.head(5000)
    # print(data.isna().sum())

    data = data[data["false counts"].notna()]

    data["true counts"].fillna(0, inplace=True)
    data.fillna("None", inplace=True)
    data.dropna(inplace=True)

    print("Full Data Shape = ", data.shape)
    # data.drop(data[data["label"] == "FALSE"].iloc[3000:].index, inplace=True)
    # print("Full Data Shape = ", data.shape)
    print(data["label"].value_counts())
    print(data["subjectivity"].value_counts())

    num_epoch = 100
    trainPoliti(data, processPoliti, politiModel,
                word2vecMatrix, num_epoch)


def main():
    liar()
    # politi()
    # handle NaN?

    # liar and politi same accuracy?


main()

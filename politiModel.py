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
from transformers import TFDistilBertModel


dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


def handleNaN(data):
    for header in data.columns.values:
        data[header] = data[header].fillna(
            data[header][data[header].first_valid_index()])
    return data


def attentionModel(metadata_vocab):
    # emb = Embedding(metadata_vocab, 128, trainable=True)

    # a1 = tf.slice(x2, [0, 0], [-1, 3])
    # a1 = x2[:,:3]
    # emb_a1 = emb(a1)
    # a2 = tf.slice(x2, [0, 3], [-1, 1])
    # a2  = x2[:]
    # emb_a2 = emb(a2)

    # att1_1 = Attention(name="att1_1")([drop1_1, emb_a1])
    # att2_1 = Attention(name="att2_1")([drop1_1, emb_a2])

    # concat1_1 = Concatenate()([drop1_1, att1_1, att2_1])
    # lstm1_1 = LSTM(32, recurrent_dropout=0.2, name="lstm1_1")(concat1_1)

    # in2 = tf.slice(x2, [0, 4], [-1, 4])
    # emb2_1 = emb(in2)

    # score = tf.slice(x2, [0, -1], [-1, 1])
    # lstm2_1 = LSTM(32, recurrent_dropout=0.2, name="lstm2_1")(emb2_1)
    # mult2_1 = Multiply()([lstm2_1, score])

    # in3 = tf.slice(x2, [0, 0], [-1, -1])
    # print(in3.shape)
    # dense3_1 = Dense(
    #     32, activation="relu", name="dense3_1")(in3)

    # x = Concatenate(name="final_concat")(
    #     [lstm1_1, mult2_1, dense3_1, dense4_1])
    # x = Multiply()([x, score])
    # x = Dense(512, activation="relu")(x)

    y1 = Dense(n_output1, activation="softmax", name="output_1")(x)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(x)

    model = keras.Model(
        inputs=[x1, x2, x3], outputs=[y1, y2])
    return model


def politiModel(seq_length, md_length, x3_length, x4_length, n_output1, n_output2):
    input_ids1 = Input(shape=(seq_length, ),
                       dtype=tf.int32, name="input_ids1")
    attention_mask1 = Input(shape=(seq_length, ),
                            dtype=tf.int32, name="attention_mask1")

    input_ids2 = Input(shape=(md_length, ),
                       dtype=tf.int32, name="input_ids2")
    attention_mask2 = Input(shape=(md_length, ),
                            dtype=tf.int32, name="attention_mask2")

    x3 = Input(shape=(x3_length,), name="x3")
    x4 = Input(shape=(x4_length,), name="x4")

    st_inputs = dict(
        input_ids=input_ids1,
        attention_mask=attention_mask1)

    md_inputs = dict(
        input_ids=input_ids2,
        attention_mask=attention_mask2,)

    bert_st = dbert_model(st_inputs)[0][:, 0, :]
    # drop_st = Dropout(0.5)(bert_st)
    # avg_st = AveragePooling1D(name="average_st")(bert_st)
    # lstm_st = LSTM(32, name="lstm_st")(avg_st)

    bert_md = dbert_model(md_inputs)[0][:, 0, :]
    # drop_md = Dropout(0.5)(bert_md)
    # avg_md = AveragePooling1D(name="average_md")(bert_md)
    # lstm_md = LSTM(32, name="lstm_md")(avg_md)

    # score = x3[:, -1]
    # print(score.shape)
    # counts = x3[:, :-2]
    # print(counts.shape)

    # dense3_1 = Dense(6, activation="relu")(counts)

    x = Concatenate()([bert_st, bert_md])
    # x = Add()([x, score])
    x = Dense(768, activation="relu")(x)

    y1 = Dense(n_output1, activation="softmax", name="output_1")(x)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(x)

    model = keras.Model(
        inputs=[{"input_ids": input_ids1,
                 "attention_mask": attention_mask1},
                {"input_ids": input_ids2,
                 "attention_mask": attention_mask2},
                x3, x4], outputs=[y1, y2])
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

    liar_train.loc[liar_train.state == "0", "state"] = "None"
    liar_test.loc[liar_test.state == "0", "state"] = "None"
    liar_val.loc[liar_val.state == "0", "state"] = "None"

    liar_train.fillna("None", inplace=True)
    liar_test.fillna("None", inplace=True)
    liar_val.fillna("None", inplace=True)

    print(liar_train["label"].value_counts())
    print(liar_train["subjectivity"].value_counts())

    num_epoch = 50
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

    num_epoch = 50
    trainPoliti(data, processPoliti, politiModel,
                word2vecMatrix, num_epoch)


def main():
    liar()
    # politi()
    # handle NaN?

    # liar and politi same accuracy?


main()

from train import train, trainLiar, trainPoliti
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertConfig

# print(dbert_model.summary())


# def handleNaN(data):
#     for header in data.columns.values:
#         data[header] = data[header].fillna(
#             data[header][data[header].first_valid_index()])
#     return data

# DistilBERT
config = DistilBertConfig(dropout=0.2,
                          attention_dropout=0.2,
                          output_hidden_states=True)
dbert_model = TFDistilBertModel.from_pretrained(
    'distilbert-base-uncased', config=config)


# Define model using Keras functional API
def buildModel(seq_length, md_length, sco_length, his_length, n_output1, n_output2):
    input_ids1 = Input(shape=(seq_length, ),
                       dtype=tf.int32, name="input_ids1")
    attention_mask1 = Input(shape=(seq_length, ),
                            dtype=tf.int32, name="attention_mask1")

    input_ids2 = Input(shape=(md_length, ),
                       dtype=tf.int32, name="input_ids2")
    attention_mask2 = Input(shape=(md_length, ),
                            dtype=tf.int32, name="attention_mask2")

    score = Input(shape=(sco_length,), name="score")
    history = Input(shape=(his_length,), name="history")

    st_inputs = dict(
        input_ids=input_ids1,
        attention_mask=attention_mask1)

    md_inputs = dict(
        input_ids=input_ids2,
        attention_mask=attention_mask2)

    dbert_st = dbert_model(st_inputs)[0]  # 3D output
    # dbert_st = dbert_model(st_inputs)[
    #     0][:, 0, :]  # 2D output, CLS only
    # drop_st = Dropout(0.2, name="dropout_st")(dbert_st)

    dbert_md = dbert_model(md_inputs)[0]  # 3D output
    # dbert_md = dbert_model(md_inputs)[
    #     0][:, 0, :]  # 2D output, CLS only
    # drop_md = Dropout(0.2, name="dropout_md")(dbert_md)

    dense_his = Dense(128, activation="relu")(history)

    x = MultiHeadAttention(num_heads=2, key_dim=2)(dbert_md, dbert_st)
    x = GlobalMaxPooling1D()(x)
    x = Add()([x, score])
    x = Concatenate()([x, dense_his])
    x = Dense(768, activation="relu")(x)
    x = Dropout(0.2)(x)

    # lower subjectivity rating with multi head
    # self multi head attention on statement data?

    y1 = Dense(n_output1, activation="softmax", name="output_1")(x)
    y2 = Dense(n_output2, activation="softmax", name="output_2")(x)

    model = keras.Model(
        inputs=[{"input_ids": input_ids1,
                 "attention_mask": attention_mask1},
                {"input_ids": input_ids2,
                 "attention_mask": attention_mask2},
                score, history], outputs=[y1, y2])
    return model


def liarEnhanced():  # Start on LIAR+ dataset
    liar_train = pd.read_csv(
        "./cleanDatasets/clean_liar_train+.csv").reset_index(drop=True)
    liar_test = pd.read_csv(
        "./cleanDatasets/clean_liar_test+.csv").reset_index(drop=True)
    liar_val = pd.read_csv(
        "./cleanDatasets/clean_liar_valid+.csv").reset_index(drop=True)

    liar_train = liar_train[liar_train["false counts"].notna()]
    liar_test = liar_test[liar_test["false counts"].notna()]

    liar_train["true counts"].fillna(0, inplace=True)
    liar_test["true counts"].fillna(0, inplace=True)
    liar_val["true counts"].fillna(0, inplace=True)

    liar_train.loc[liar_train.state == "0", "state"] = "None"
    liar_test.loc[liar_test.state == "0", "state"] = "None"
    liar_val.loc[liar_val.state == "0", "state"] = "None"

    liar_train.fillna("None", inplace=True)
    liar_test.fillna("None", inplace=True)
    liar_val.fillna("None", inplace=True)

    print(liar_train["label"].value_counts())
    print(liar_train["subjectivity"].value_counts())

    num_epoch = 6
    trainLiar(liar_train, liar_test, liar_val, buildModel, num_epoch)


def liar():  # Start on LIAR dataset
    liar_train = pd.read_csv(
        "./cleanDatasets/clean_liar_train.csv").reset_index(drop=True)
    liar_test = pd.read_csv(
        "./cleanDatasets/clean_liar_test.csv").reset_index(drop=True)
    liar_val = pd.read_csv(
        "./cleanDatasets/clean_liar_valid.csv").reset_index(drop=True)

    liar_train = liar_train[liar_train["false counts"].notna()]
    liar_test = liar_test[liar_test["false counts"].notna()]

    liar_train.loc[liar_train.state == "0", "state"] = "None"
    liar_test.loc[liar_test.state == "0", "state"] = "None"
    liar_val.loc[liar_val.state == "0", "state"] = "None"

    liar_train.fillna("None", inplace=True)
    liar_test.fillna("None", inplace=True)
    liar_val.fillna("None", inplace=True)

    print(liar_train["label"].value_counts())
    print(liar_train["subjectivity"].value_counts())

    num_epoch = 20
    trainLiar(liar_train, liar_test, liar_val, buildModel, num_epoch)


def politi():  # Start on POLITI dataset
    data = pd.read_csv(
        "./cleanDatasets/politi+.csv").reset_index(drop=True)

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

    num_epoch = 20
    trainPoliti(data, buildModel, num_epoch)


def main():
    liar()
    # politi()


main()

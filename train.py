import pandas as pd
import numpy as np
import keras
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tokenization import process

optimizer = tf.keras.optimizers.Adam()
optimizer_ft = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=5e-5)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

batch_size = 16


def plot(history):  # Plot accuracy and loss
    # "Accuracy"
    plt.plot(history.history["output_1_accuracy"])
    plt.plot(history.history["output_2_accuracy"])
    plt.plot(history.history["val_output_1_accuracy"])
    plt.plot(history.history["val_output_2_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Output1 Train", "Output2 Train",
                "Output1 Validation", "Output2 Validation"], loc="lower left")
    plt.show()
    # "Loss"
    plt.plot(history.history["output_1_loss"])
    plt.plot(history.history["output_2_loss"])
    plt.plot(history.history["val_output_1_loss"])
    plt.plot(history.history["val_output_2_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Output1 Train", "Output2 train",
                "Output1 Validation", "Output2 Validation"], loc="lower left")
    plt.show()


# Train model
def train(x_train,  y_train, x_test, y_test, x_val, y_val, createModelFunction, num_epoch):
    x_train[3], x_test[3], x_val[3] = normalize(
        x_train[3]), normalize(x_test[3]), normalize(x_val[3])

    print("input_ids_train1 Shape", x_train[0]["input_ids"].shape)
    print("attention_mask_train1 Shape", x_train[0]["attention_mask"].shape)
    print("tokens_train2 Shape", x_train[1]["input_ids"].shape)
    print("masks_train2 Shape", x_train[1]["attention_mask"].shape)
    print("x_train3 Shape", x_train[2].shape)
    print("x_train4 Shape", x_train[3].shape)

    print("input_ids_test1 Shape", x_test[0]["input_ids"].shape)
    print("attention_mask_test1 Shape", x_test[0]["attention_mask"].shape)
    print("input_ids_test2 Shape", x_test[1]["input_ids"].shape)
    print("attention_mask_test2 Shape", x_test[1]["attention_mask"].shape)
    print("x_test3 Shape", x_test[2].shape)
    print("x_test4 Shape", x_test[3].shape)

    print("input_ids_val1 Shape", x_val[0]["input_ids"].shape)
    print("attention_mask_val1 Shape", x_val[0]["attention_mask"].shape)
    print("input_ids_val2 Shape", x_val[1]["input_ids"].shape)
    print("attention_mask_val2 Shape", x_val[1]["attention_mask"].shape)
    print("x_val3 Shape", x_val[2].shape)
    print("x_val4 Shape", x_val[3].shape)

    print("y_train1 Shape", y_train[0].shape)
    print("y_train2 Shape", y_train[1].shape)
    print("y_test1 Shape", y_test[0].shape)
    print("y_test2 Shape", y_test[1].shape)
    print("y_val1 Shape", y_val[0].shape)
    print("y_val2 Shape", y_val[1].shape)

    seq_length = x_train[0]["input_ids"].shape[1]
    md_length = x_train[1]["input_ids"].shape[1]
    sco_length = x_train[2].shape[1]
    his_length = x_train[3].shape[1]

    n_output1 = y_train[0].shape[1]
    n_output2 = y_train[1].shape[1]

    model = createModelFunction(
        seq_length, md_length, sco_length, his_length, n_output1, n_output2)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_output_1_loss', patience=3)

    # Freeze DBERT layers
    for layer in model.layers:
        if layer.name == "tf_distil_bert_model":
            layer.trainable = False

    model.summary()
    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), callbacks=[early_stop], epochs=num_epoch,  batch_size=batch_size, verbose=1)

    print("### EVALUATION ###")
    model.evaluate(x_test, y_test, verbose=1)
    plot(history)

    # Fine Tuning DBERT, unfreeze layers
    for layer in model.layers:
        if layer.name == "tf_distil_bert_model":
            layer.trainable = True

    model.summary()
    model.compile(loss=loss_function,
                  optimizer=optimizer_ft, metrics=["accuracy"])

    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), callbacks=[early_stop], epochs=num_epoch,  batch_size=batch_size, verbose=1)

    print("### EVALUATION ###")
    model.evaluate(x_test, y_test, verbose=1)
    plot(history)


# Handle train, test, valid for LIAR
def trainLiar(liar_train, liar_test, liar_val, createModelFunction, num_epoch):
    liar_train = shuffle(liar_train, random_state=42)
    liar_test = shuffle(liar_test, random_state=42)
    liar_val = shuffle(liar_val, random_state=42)

    x_train, y_train = process(liar_train)
    x_test, y_test = process(liar_test)
    x_val, y_val = process(liar_val)

    train(x_train,  y_train, x_test, y_test, x_val,
          y_val, createModelFunction, num_epoch)


# Handle train, test, valid for POLITI
def trainPoliti(data, createModelFunction, num_epoch):
    # data = shuffle(data, random_state=42)
    p_train, p_test = train_test_split(
        data, test_size=0.1, random_state=42, shuffle=True, stratify=data["label"])
    p_train, p_val = train_test_split(
        p_train, test_size=0.112, random_state=42, shuffle=True, stratify=p_train["label"])

    x_train, y_train = process(p_train)
    x_test, y_test = process(p_test)
    x_val, y_val = process(p_val)

    train(x_train,  y_train, x_test, y_test, x_val,
          y_val, createModelFunction, num_epoch)

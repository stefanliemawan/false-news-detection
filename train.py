import pandas as pd
import numpy as np
import keras
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# optimizer = tf.keras.optimizers.SGD()
# optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
optimizer = tfa.optimizers.AdamW(weight_decay=0.0, learning_rate=3e-5)
# optimizer = tfa.optimizers.AdamW(weight_decay=0.3, learning_rate=3e-3)
# optimizer = tf.keras.optimizers.RMSprop()
# optimizer = tf.keras.optimizers.Adadelta()
# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_multiclass = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_binary = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# loss_regression = tf.keras.losses.MeanAbsoluteError()
# cat_accuracy = tf.keras.metrics.CategoricalAccuracy(
#     name="cat_accuracy", dtype=None)

batch_size = 16

# batch_size 16 and lr 3e-5


class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data

    # def on_train_batch_begin(self, batch, logs=None):
    #     print('Training: batch {} begins at {}'.format(
    #         batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            self.model.evaluate(self.validation_data, verbose=2)
            print("\n")
        # print('Training: batch {} ends at {}'.format(
        #     batch, datetime.datetime.now().time()))
        # print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch {} ends at {}'.format(
            epoch, datetime.datetime.now().time()))


def plot(history):
    # print(history.history.keys())
    # "Accuracy"
    plt.plot(history.history["output_1_accuracy"])
    plt.plot(history.history["output_2_accuracy"])
    plt.plot(history.history["val_output_1_accuracy"])
    plt.plot(history.history["val_output_2_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Output1 Train", "Output2 Train",
                "Output1 Validation", "Output2 Validation"], loc="lower right")
    plt.show()
    # "Loss"
    # plt.plot(history.history["output_1_loss"])
    # plt.plot(history.history["output_2_loss"])
    # plt.plot(history.history["val_output_1_loss"])
    # plt.plot(history.history["val_output_2_loss"])
    # plt.title("Model Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["Output1 Train", "Output2 train",
    #             "Output1 Validation", "Output2 Validation"], loc="lower right")
    # plt.show()


def train(x_train1, x_train2, x_train3, x_train4, x_test1, x_test2, x_test3, x_test4, x_val1, x_val2, x_val3, x_val4, y_train1,
          y_test1, y_val1, y_train2, y_test2, y_val2, createModelFunction, num_epoch):
    x_train3, x_test3, x_val3 = normalize(
        x_train3), normalize(x_test3), normalize(x_val3)
    # x_train4, x_test4, x_val4 = normalize(
    #     x_train4), normalize(x_test4), normalize(x_val4)

    n_output1 = y_train1.shape[1]
    n_output2 = y_train2.shape[1]

    print("input_ids_train1 Shape", x_train1["input_ids"].shape)
    print("attention_mask_train1 Shape", x_train1["attention_mask"].shape)
    print("tokens_train2 Shape", x_train2["input_ids"].shape)
    print("masks_train2 Shape", x_train2["attention_mask"].shape)
    print("x_train3 Shape", x_train3.shape)
    print("x_train4 Shape", x_train4.shape)

    print("input_ids_test1 Shape", x_test1["input_ids"].shape)
    print("attention_mask_test1 Shape", x_test1["attention_mask"].shape)
    print("input_ids_test2 Shape", x_test2["input_ids"].shape)
    print("attention_mask_test2 Shape", x_test2["attention_mask"].shape)
    print("x_test3 Shape", x_test3.shape)
    print("x_test4 Shape", x_test4.shape)

    print("input_ids_val1 Shape", x_val1["input_ids"].shape)
    print("attention_mask_val1 Shape", x_val1["attention_mask"].shape)
    print("input_ids_val2 Shape", x_val2["input_ids"].shape)
    print("attention_mask_val2 Shape", x_val2["attention_mask"].shape)
    print("x_val3 Shape", x_val3.shape)
    print("x_val4 Shape", x_val4.shape)

    print("y_train1 Shape", y_train1.shape)
    print("y_train2 Shape", y_train2.shape)
    print("y_test1 Shape", y_test1.shape)
    print("y_test2 Shape", y_test2.shape)
    print("y_val1 Shape", y_val1.shape)
    print("y_val2 Shape", y_val2.shape)

    seq_length = x_train1["input_ids"].shape[1]
    md_length = x_train2["input_ids"].shape[1]

    model = createModelFunction(
        seq_length, md_length, x_train3.shape[1], x_train4.shape[1], n_output1, n_output2)
    model.summary()

    model.compile(loss=[loss_multiclass, loss_multiclass],
                  optimizer=optimizer, metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)  # 3

    x_train = [{"input_ids": x_train1["input_ids"],
                "attention_mask": x_train1["attention_mask"]},
               {"input_ids": x_train2["input_ids"],
                "attention_mask": x_train2["attention_mask"]},
               x_train3, x_train4]

    x_val = [{"input_ids": x_val1["input_ids"],
              "attention_mask": x_val1["attention_mask"]},
             {"input_ids": x_val2["input_ids"],
              "attention_mask": x_val2["attention_mask"]}, x_val3, x_val4]

    x_test = [{"input_ids": x_test1["input_ids"],
               "attention_mask": x_test1["attention_mask"]},
              {"input_ids": x_test2["input_ids"],
               "attention_mask": x_test2["attention_mask"]}, x_test3, x_test4]

    history = model.fit(
        x_train, [y_train1, y_train2], validation_data=(x_val, [y_val1, y_val2]), callbacks=[early_stop], epochs=num_epoch,  batch_size=batch_size, verbose=1)

    # cross validation?

    print("### EVALUATION ###")
    model.evaluate(x_test, [y_test1, y_test2], verbose=1)
    plot(history)

    # model.save("./models/10k-128kimcnn-100lstm-model1.h5")


def trainLiar(liar_train, liar_test, liar_val, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    liar_train = shuffle(liar_train, random_state=1)
    liar_test = shuffle(liar_test, random_state=1)
    liar_val = shuffle(liar_val, random_state=1)

    x_train1, x_train2, x_train3, x_train4, y_train1, y_train2 = processFunction(
        liar_train)
    x_test1, x_test2, x_test3, x_test4, y_test1, y_test2 = processFunction(
        liar_test)
    x_val1, x_val2, x_val3, x_val4, y_val1, y_val2 = processFunction(
        liar_val)

    train(x_train1, x_train2, x_train3, x_train4, x_test1, x_test2, x_test3, x_test4, x_val1, x_val2, x_val3, x_val4, y_train1,
          y_test1, y_val1, y_train2, y_test2, y_val2, createModelFunction, num_epoch)


def trainPoliti(data, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    # data = shuffle(data, random_state=42)

    p_train, p_test = train_test_split(
        data, test_size=0.1, random_state=42, shuffle=True, stratify=data["label"])
    p_train, p_val = train_test_split(
        p_train, test_size=0.112, random_state=42, shuffle=True, stratify=p_train["label"])

    x_train1, x_train2, x_train3, x_train4, y_train1, y_train2 = processFunction(
        p_train)
    x_val1, x_val2, x_val3, x_val4, y_val1, y_val2 = processFunction(p_val)
    x_test1, x_test2, x_test3, x_test4, y_test1, y_test2 = processFunction(
        p_test)

    train(x_train1, x_train2, x_train3, x_train4,  x_test1, x_test2, x_test3, x_test4,  x_val1, x_val2, x_val3, x_val4, y_train1, y_test1,
          y_val1, y_train2, y_test2, y_val2, createModelFunction, num_epoch)

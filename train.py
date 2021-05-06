import pandas as pd
import numpy as np
import keras
import tensorflow as tf
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
from tokenization import returnStatementMaxlen

# optimizer = tf.keras.optimizers.SGD()
optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.RMSprop()
# optimizer = tf.keras.optimizers.Adadelta()
# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_multiclass = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_binary = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# loss_regression = tf.keras.losses.MeanAbsoluteError()
batch_size = 64


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
    plt.plot(history.history["output_1_loss"])
    plt.plot(history.history["output_2_loss"])
    plt.plot(history.history["val_output_1_loss"])
    plt.plot(history.history["val_output_2_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Output1 Train", "Output2 train",
                "Output1 Validation", "Output2 Validation"], loc="lower right")
    plt.show()


def train(x_train1, x_train2, x_train3, x_train4,  x_test1, x_test2, x_test3, x_test4,  x_val1, x_val2, x_val3, x_val4, y_train1, y_test1, y_val1, y_train2, y_test2, y_val2, createModelFunction, num_epoch):
    x_train3, x_test3, x_val3 = normalize(
        x_train3), normalize(x_test3), normalize(x_val3)
    x_train4, x_test4, x_val4 = normalize(
        x_train4), normalize(x_test4), normalize(x_val4)

    # statement_vocab = emb_matrix.shape[0]
    metadata_vocab = round(np.max(x_train2))+1
    # print("Statement Vocab", statement_vocab)
    print("Metadata Vocab", metadata_vocab)

    tokens_train = x_train1[0]
    masks_train = x_train1[1]
    segments_train = x_train1[2]

    tokens_test = x_test1[0]
    masks_test = x_test1[1]
    segments_test = x_test1[2]

    tokens_val = x_val1[0]
    masks_val = x_val1[1]
    segments_val = x_val1[2]

    seq_length = returnStatementMaxlen()

    n_output1 = y_train1.shape[1]
    n_output2 = y_train2.shape[1]
    # n_output2 = 1

    print("tokens_train Shape", tokens_train.shape)
    print("masks_train Shape", masks_train.shape)
    print("segments_train Shape", segments_train.shape)
    print("x_train2 Shape", x_train2.shape)
    print("x_train3 Shape", x_train3.shape)
    print("x_train4 Shape", x_train4.shape)

    print("tokens_test Shape", tokens_test.shape)
    print("masks_test Shape", masks_test.shape)
    print("segments_test Shape", segments_test.shape)
    print("x_test2 Shape", x_test2.shape)
    print("x_test3 Shape", x_test3.shape)
    print("x_test4 Shape", x_test4.shape)

    print("tokens_val Shape", tokens_val.shape)
    print("masks_val Shape", masks_val.shape)
    print("segments_val Shape", segments_val.shape)
    print("x_val2 Shape", x_val2.shape)
    print("x_val3 Shape", x_val3.shape)
    print("x_val4 Shape", x_val4.shape)

    print("y_train1 Shape", y_train1.shape)
    print("y_train2 Shape", y_train2.shape)

    print("y_test1 Shape", y_test1.shape)
    print("y_test2 Shape", y_test2.shape)

    print("y_val1 Shape", y_val1.shape)
    print("y_val2 Shape", y_val2.shape)

    model = createModelFunction(
        seq_length, x_train2.shape, x_train3.shape, x_train4.shape, metadata_vocab, n_output1, n_output2)
    model.summary()

    model.compile(loss=loss_multiclass,
                  optimizer=optimizer, metrics=["accuracy"])

    # callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=3)  # 3

    # history = model.fit(
    #     [x_train1, x_train2, x_train3, x_train4], [y_train1, y_train2], epochs=num_epoch, validation_data=([x_val1, x_val2, x_val3, x_val4], [y_val1, y_val2]),  callbacks=[callback], batch_size=batch_size, verbose=1)

    history = model.fit(
        {"input_word_ids": tokens_train, "input_mask": masks_train, "input_type_ids": segments_train, "x2": x_train2, "x3": x_train3, "x4": x_train4}, [y_train1, y_train2], epochs=num_epoch, validation_data=({"input_word_ids": tokens_val, "input_mask": masks_val, "input_type_ids": segments_val, "x2": x_val2, "x3": x_val3, "x4": x_val4}, [y_val1, y_val2]), batch_size=batch_size, verbose=1)

    # cross validation?

    # print("### EVALUATION ###")
    # model.evaluate([tokens_test, masks_test, segments_test, x_test2, x_test3, x_test4],
    #                [y_test1, y_test2], verbose=1)
    # plot(history)

    # model.save("./models/10k-128kimcnn-100lstm-model1.h5")


def trainLiar(liar_train, liar_test, liar_val, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    liar_train = shuffle(liar_train, random_state=42)
    liar_test = shuffle(liar_test, random_state=42)
    liar_val = shuffle(liar_val, random_state=42)

    x_train1, x_train2, x_train3, x_train4, y_train1, y_train2 = processFunction(
        liar_train)
    x_val1, x_val2, x_val3, x_val4, y_val1, y_val2 = processFunction(liar_val)
    x_test1, x_test2, x_test3, x_test4, y_test1, y_test2 = processFunction(
        liar_test)

    # emb_matrix = createEmbeddingFunction(liar_train["statement"])
    # print("Embedding Matrix Shape", emb_matrix.shape)

    train(x_train1, x_train2, x_train3, x_train4,  x_test1, x_test2, x_test3, x_test4,  x_val1, x_val2, x_val3, x_val4, y_train1,
          y_test1, y_val1, y_train2, y_test2, y_val2, createModelFunction, num_epoch)


def trainPoliti(data, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    # data = shuffle(data, random_state=42)

    p_train, p_test = train_test_split(
        data, test_size=0.1, random_state=42, shuffle=True, stratify=data["label"])
    p_train, p_val = train_test_split(
        p_train, test_size=0.11, random_state=42, shuffle=True, stratify=p_train["label"])

    x_train1, x_train2, x_train3, x_train4, y_train1, y_train2 = processFunction(
        p_train)
    x_val1, x_val2, x_val3, x_val4, y_val1, y_val2 = processFunction(p_val)
    x_test1, x_test2, x_test3, x_test4, y_test1, y_test2 = processFunction(
        p_test)

    emb_matrix = createEmbeddingFunction(p_train["statement"])
    print("Embedding Matrix Shape", emb_matrix.shape)

    train(x_train1, x_train2, x_train3, x_train4,  x_test1, x_test2, x_test3, x_test4,  x_val1, x_val2, x_val3, x_val4, y_train1, y_test1,
          y_val1, y_train2, y_test2, y_val2, emb_matrix, createEmbeddingFunction, createModelFunction, num_epoch)

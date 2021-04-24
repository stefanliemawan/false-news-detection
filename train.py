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
# from tokenization import returnVocabSize

# optimizer = tf.keras.optimizers.SGD()
optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.RMSprop()
# optimizer = tf.keras.optimizers.Adadelta()
# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# loss_classification = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# loss_regression = tf.keras.losses.MeanAbsoluteError()
batch_size = 64


def plot(history):
    # print(history.history.keys())
    plt.plot(history.history["output_1_accuracy"])
    plt.plot(history.history["output_2_accuracy"])
    plt.plot(history.history["val_output_1_accuracy"])
    plt.plot(history.history["val_output_2_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Output1 Train", "Output2 Train",
                "Output1 Validation", "Output2 Validation"], loc="upper left")
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
    #             "Output1 Validation", "Output2 Validation"], loc="upper left")
    # plt.show()


def train(x_train1, x_train2, x_train3,  x_test1, x_test2, x_test3,  x_val1, x_val2, x_val3, y_train1, y_train2, y_test1, y_test2, y_val1, y_val2, emb_matrix, createEmbeddingFunction, createModelFunction, num_epoch):

    statement_vocab = emb_matrix.shape[0]
    metadata_vocab = round(np.max(x_train2)+1)
    print("Statement Vocab", statement_vocab)
    print("Metadata Vocab", metadata_vocab)

    n_output1 = y_train1.shape[1]
    n_output2 = y_train2.shape[1]

    print("x_train1 Shape", x_train1.shape)
    print("x_train2 Shape", x_train2.shape)
    print("x_train3 Shape", x_train3.shape)
    print("y_train1 Shape", y_train1.shape)
    print("y_train2 Shape", y_train2.shape)
    print("x_test1 Shape", x_test1.shape)
    print("x_test2 Shape", x_test2.shape)
    print("x_test3 Shape", x_test3.shape)
    print("y_test1 Shape", y_test1.shape)
    print("y_test2 Shape", y_test2.shape)
    print("x_val1 Shape", x_val1.shape)
    print("x_val2 Shape", x_val2.shape)
    print("x_val3 Shape", x_val3.shape)
    print("y_val1 Shape", y_val1.shape)
    print("y_val2 Shape", y_val2.shape)

    model = createModelFunction(
        x_train1.shape, x_train2.shape, x_train3.shape, statement_vocab, metadata_vocab, n_output1, n_output2, emb_matrix)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(
        [x_train1, x_train2, x_train3], [y_train1, y_train2], epochs=num_epoch, validation_data=([x_val1, x_val2, x_val3], [y_val1, y_val2]), batch_size=batch_size, verbose=1)

    print("### EVALUATION ###")
    model.evaluate([x_test1, x_test2, x_test3], [y_test1, y_test2], verbose=1)
    model.evaluate([x_val1, x_val2, x_val3], [y_val1, y_val2], verbose=1)
    plot(history)

    # model.save("./models/10k-128kimcnn-100lstm-model1.h5")


def trainLiar(liar_train, liar_test, liar_val, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    liar_train = shuffle(liar_train)
    liar_test = shuffle(liar_test)
    liar_val = shuffle(liar_val)

    x_train1, x_train2, y_train1, y_train2 = processFunction(liar_train)
    x_test1, x_test2, y_test1, y_test2 = processFunction(liar_test)
    x_val1, x_val2, y_val1, y_val2 = processFunction(liar_val)

    emb_matrix = createEmbeddingFunction(liar_train["statement"])
    print("Embedding Matrix Shape", emb_matrix.shape)

    train(x_train1, x_train2, x_train3,  x_test1, x_test2, x_test3,  x_val1, x_val2, x_val3, y_train1, y_train2,
          y_test1, y_test2, y_val1, y_val2, emb_matrix, createEmbeddingFunction, createModelFunction, num_epoch)


def trainPoliti(data, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    data = shuffle(data)

    x1, x2, x3, y1, y2 = processFunction(data)

    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x1, y1, test_size=0.1, random_state=42)
    x_train1, x_val1, y_train1, y_val1 = train_test_split(
        x_train1, y_train1, test_size=0.11, random_state=42)

    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        x2, y2, test_size=0.1, random_state=42)
    x_train2, x_val2, y_train2, y_val2 = train_test_split(
        x_train2, y_train2, test_size=0.11, random_state=42)

    x_train3, x_test3 = train_test_split(x3, test_size=0.1, random_state=42)
    x_train3, x_val3 = train_test_split(
        x_train3, test_size=0.11, random_state=42)

    emb_matrix = createEmbeddingFunction(data["statement"])
    print("Embedding Matrix Shape", emb_matrix.shape)

    train(x_train1, x_train2, x_train3,  x_test1, x_test2, x_test3,  x_val1, x_val2, x_val3, y_train1, y_train2,
          y_test1, y_test2, y_val1, y_val2, emb_matrix, createEmbeddingFunction, createModelFunction, num_epoch)

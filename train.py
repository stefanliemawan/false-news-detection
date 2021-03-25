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
from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# optimizer = tf.keras.optimizers.SGD()
optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.RMSprop()
# optimizer = tf.keras.optimizers.Adadelta()
# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
batch_size = 64


def plot(history):
    # print(history.history.keys())
    plt.plot(history.history['output_1_accuracy'])
    plt.plot(history.history['output_2_accuracy'])
    plt.plot(history.history['val_output_1_accuracy'])
    plt.plot(history.history['val_output_2_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Output1 Train', 'Output2 Train',
                'Output1 Validation', 'Output2 Validation'], loc='upper left')
    plt.show()
    # "Loss"
    # plt.plot(history.history['output_1_loss'])
    # plt.plot(history.history['output_2_loss'])
    # plt.plot(history.history['val_output_1_loss'])
    # plt.plot(history.history['val_output_2_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Output1 Train', 'Output2 train',
    #             'Output1 Validation', 'Output2 Validation'], loc='upper left')
    # plt.show()


def train(data, processFunction, createModelFunction, createEmbeddingFunction, num_epoch):
    data = shuffle(data)
    # data = data.head(5000)
    emb_matrix = createEmbeddingFunction(data['statement'])

    x1, x2, y1, y2 = processFunction(data)

    x2 = normalize(x2)

    n_output1 = y1.shape[1]
    n_output2 = y2.shape[1]

    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x1, y1, test_size=0.09, random_state=42)
    x_train1, x_val1, y_train1, y_val1 = train_test_split(
        x_train1, y_train1, test_size=0.1, random_state=42)

    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        x2, y2, test_size=0.09, random_state=42)
    x_train2, x_val2, y_train2, y_val2 = train_test_split(
        x_train2, y_train2, test_size=0.1, random_state=42)

    print('x_train1 shape =', x_train1.shape)
    print('x_train2 shape =', x_train2.shape)
    print('y_train1 shape =', y_train1.shape)
    print('y_train2 shape =', y_train2.shape)
    print('x_test1 shape =', x_test1.shape)
    print('x_test2 shape =', x_test2.shape)
    print('y_test1 shape =', y_test1.shape)
    print('y_test2 shape =', y_test2.shape)
    print('x_val1 shape =', x_val1.shape)
    print('x_val2 shape =', x_val2.shape)
    print('y_val1 shape =', y_val1.shape)
    print('y_val2 shape =', y_val2.shape)

    model = createModelFunction(
        x_train1.shape, x_train2.shape, n_output1, n_output2, emb_matrix)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(
        [x_train1, x_train2], [y_train1, y_train2], epochs=num_epoch, validation_data=([x_val1, x_val2], [y_val1, y_val2]), batch_size=batch_size, verbose=1)

    # kf = KFold(n_splits=5, shuffle=True)
    # k_fold = 1

    # for train_index, test_index in kf.split(x_train1):
    #     print("k = ", k_fold)

    #     k_x_train1, k_x_test1 = x_train1[train_index], x_train1[test_index]
    #     k_x_train2, k_x_test2 = x_train2[train_index], x_train2[test_index]
    #     k_y_train1, k_y_test1 = y_train1[train_index], y_train1[test_index]
    #     k_y_train2, k_y_test2 = y_train2[train_index], y_train2[test_index]
    #     model.fit(
    #         [k_x_train1, k_x_train2], [k_y_train1, k_y_train2], epochs=num_epoch, validation_data=([k_x_test1, k_x_test2], [k_y_test1, k_y_test2]), batch_size=batch_size, verbose=1)
    #     model.evaluate([k_x_test1, k_x_test2], [
    #                    k_y_test1, k_y_test2], verbose=1)
    #     # print("### EVALUATION ###")
    #     # model.evaluate([x_test1, x_test2], [y_test1, y_test2], verbose=1)
    #     # model.evaluate([x_val1, x_val2], [y_val1, y_val2], verbose=1)

    #     k_fold += 1

    print("### EVALUATION ###")
    model.evaluate([x_test1, x_test2], [y_test1, y_test2], verbose=1)
    model.evaluate([x_val1, x_val2], [y_val1, y_val2], verbose=1)
    plot(history)

    # model.save('./models/128-20-no-duplicate-model1.h5')

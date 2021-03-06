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


from tokenization import handleNaN, process, glove

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Bi LSTM, Scikit Learn, Keras
# Mixed data neural network
# http: // digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/

# vocab_size = 10375  # use maxlen?
# vocab_size = 13304  # use maxlen?
vocab_size = 0  # use maxlen?
embedding_dim = 300
batch_size = 64

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.RMSprop()
# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


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
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()


def createModel(x1_shape, x2_shape, n_output1, n_output2):
    # 1 not learning enough, stable at 0.2
    # 2 not learning anything
    x1 = Input(shape=(x1_shape[1], x1_shape[2]), name="input_1")
    cnn1_1 = Conv1D(128, 5,
                    activation="relu", name="conv1d_1_1")(x1)
    gmp1_1 = GlobalMaxPooling1D(name="global_max_pooling1D_1")(cnn1_1)

    x2 = Input(shape=(x2_shape[1], 1), name="input_2")
    cnn2_1 = Conv1D(64, 5,
                    activation="relu", name="conv1d_2_1")(x2)
    bilstm2_1 = Bidirectional(LSTM(32, name="bi_lstm_2_1"))(cnn2_1)

    x = concatenate([gmp1_1, bilstm2_1])
    x = Dense(12, activation="relu")(x)

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    # change subjectivity into regression? y2
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def testModel1(x_shape, n_output1):
    # overfit
    # word2vec?
    x1 = Input(shape=(x_shape[1], x_shape[2]), name="input_1")
    cnn1 = Conv1D(128, 5,
                  activation="relu", kernel_regularizer="l2", name="conv1d_1")(x1)
    drop1 = Dropout(0.2, name="dropout_1")(cnn1)
    bn1 = BatchNormalization(name="batch_normalization_1")(drop1)
    mp1 = MaxPooling1D(3, name="max_pooling1d_1")(bn1)
    flat1 = Flatten(name="flat_1")(mp1)
    dense1 = Dense(64,
                   activation="relu", name="dense_1")(flat1)
    drop2 = Dropout(0.2, name="dropout_2")(dense1)
    y1 = Dense(n_output1, activation='softmax', name="output_1")(drop2)
    model = keras.Model(inputs=x1, outputs=y1)
    return model


def train():
    liar_train = handleNaN(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    liar_test = handleNaN(pd.read_csv('./cleanDatasets/clean_liar_test.csv'))
    liar_valid = handleNaN(pd.read_csv('./cleanDatasets/clean_liar_valid.csv'))

    liar = pd.concat([liar_train, liar_test, liar_valid])
    liar = shuffle(liar)

    x1, x2, y1, y2 = process(liar)

    x2 = normalize(x2)

    n_output1 = y1.shape[1]
    n_output2 = y2.shape[1]

    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x1, y1, test_size=0.1, random_state=42)
    x_train1, x_val1, y_train1, y_val1 = train_test_split(
        x_train1, y_train1, test_size=0.1, random_state=42)

    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        x2, y2, test_size=0.1, random_state=42)
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

    model = createModel(
        x_train1.shape, x_train2.shape, n_output1, n_output2)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 20

    history = model.fit(
        [x_train1, x_train2], [y_train1, y_train2], epochs=num_epochs, validation_data=([x_val1, x_val2], [y_val1, y_val2]), batch_size=batch_size, verbose=1)
    plot(history)

    # kf = KFold(n_splits=10, shuffle=True)
    # k_fold = 1

    # for train_index, test_index in kf.split(x_train1):
    #     print("k = ", k_fold)

    #     k_x_train2, k_x_test2 = x_train2[train_index], x_train2[test_index]
    #     k_y_train2, k_y_test2 = y_train2[train_index], y_train2[test_index]
    #     model.fit(
    #         [k_x_train1, k_x_train2], [k_y_train1, k_y_train2], epochs=num_epochs, validation_data=([k_x_test1, k_x_test2], [k_y_test1, k_y_test2]), batch_size=batch_size, verbose=1)
    #     model.evaluate([k_x_test1, k_x_test2], [
    #                    k_y_test1, k_y_test2], verbose=1)

    #     k_fold += 1

    model.evaluate([x_test1, x_test2], [y_test1, y_test2], verbose=1)
    model.evaluate([x_val1, x_val2], [y_val1, y_val2], verbose=1)


def main():
    train()


main()

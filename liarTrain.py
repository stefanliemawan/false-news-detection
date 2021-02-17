import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from tokenization import normalize, process1, process2, process3, returnStatementTokenizer
from keras.optimizers import SGD

# Bi LSTM, Scikit Learn, Keras
# Mixed data neural network
# http: // digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/

vocab_size = 11000  # use maxlen?
# vocab_size = 13304  # use maxlen?
embedding_dim = 100
batch_size = 32

optimizer = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def glove(texts):
    tokenizer = returnStatementTokenizer()
    tokenizer.fit_on_texts(texts.values)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    emb_index = {}
    with open("./glove/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            emb_index[word] = vector
    print('Loaded %s word vectors.' % len(emb_index))

    emb_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        emb_vector = emb_index.get(word)
        if emb_vector is not None:
            # words not found in embedding index will be all-zeros.
            emb_matrix[i] = emb_vector

    return emb_matrix


def createModel1(n_output, emb_matrix):
    model = Sequential()
    model.add(Input(shape=(50,)))
    model.add(Embedding(vocab_size, embedding_dim, weights=[emb_matrix]))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output, activation="softmax"))
    return model


def train1():
    liar_train = pd.read_csv(
        './cleanDatasets/clean_liar_train.csv')
    # liar_train = shuffle(liar_train.reset_index(drop=True))
    emb_matrix = glove(liar_train['statement'])
    x_train, y_train = process1(liar_train)

    print("emb_matrix = ", emb_matrix.shape)
    print("x_train = ", x_train.shape)
    print("y_train = ", y_train.shape)

    n_output = max(y_train)+1

    model = createModel1(n_output, emb_matrix)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 10
    history = model.fit(
        x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    # 1 - 100 epoch - loss: 1.1556 - accuracy: 0.8879 (without pre trained glove)
    # 2 - 100 epoch - loss: 1.2017 - accuracy: 0.8416 (glove.6B.100d)

#  sucks find out why
#  goes convergence at low acc after several epoch


def createModel2(xseq_shape, xnum_shape, n_output):
    seq_input = Input(shape=(xseq_shape[1],), name='seq_input')
    print(seq_input.shape)
    seq = Embedding(vocab_size, embedding_dim)(seq_input)
    # seq = Bidirectional(LSTM(128, dropout=0.3,
    #                          recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(seq)
    seq = Bidirectional(LSTM(128, dropout=0.3,
                             recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(seq)
    print(seq.shape)

    num_input = Input(shape=(xnum_shape[1],), name='num_input')
    num = Dense(128)(num_input)
    print(num_input.shape)

    x = concatenate([seq, num])
    x = Dense(64, activation='relu')(x)
    x = Dense(n_output, activation='softmax')(x)
    model = keras.Model(inputs=[seq_input, num_input], outputs=[x])
    return model


def train2():
    liar_train = normalize(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    # liar_train = shuffle(liar_train.reset_index(drop=True))
    # emb_matrix = glove(liar_train['statement'])
    xseq_train, xnum_train, y_train = process2(liar_train)

    # print("emb_matrix = ", emb_matrix.shape)

    n_output = max(y_train)+1

    xseq_train = np.asarray(xseq_train, dtype=np.float)
    xnum_train = np.asarray(xnum_train, dtype=np.float)
    y_train = np.asarray(y_train, dtype=np.float)

    print('xseq_train shape = ', xseq_train.shape)
    print('xnum_train shape = ', xnum_train.shape)
    print('y_train shape = ', y_train.shape)

    model = createModel2(
        xseq_train.shape, xnum_train.shape, n_output)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 50

    history = model.fit(
        [xseq_train, xnum_train], y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    # 10 epoch -  loss: 1.8330 - accuracy: 0.2106 stable after 6th epoch


def createModel3(x_length, n_output):
    model = Sequential()
    model.add(Input(shape=(x_length,)))
    # model.add(Embedding(vocab_size, embedding_dim, weights=[emb_matrix]))
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output, activation="softmax"))
    return model


def train3():
    liar_train = normalize(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    liar_test = normalize(pd.read_csv('./cleanDatasets/clean_liar_test.csv'))
    # emb_matrix = glove(liar_train['statement'])
    print('Processing Training Data...')
    x_train, y_train = process3(liar_train)

    print('Processing Test Data...')
    x_test, y_test = process3(liar_test)

    # x_test = np.reshape(x_test, (-1, 77))

    x_train = np.asarray(x_train, dtype=np.float)
    x_test = np.asarray(x_test, dtype=np.float)

    print('x_train shape =', x_train.shape)
    print('y_train shape =', y_train.shape)
    print('x_test shape =', x_test.shape)
    print('y_test shape =', y_test.shape)

    n_output = max(y_train)+1

    model = createModel3(x_train.shape[1], n_output)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 1
    history = model.fit(
        x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    result = model.evaluate(x_test, y_test, verbose=1)
    print(result)

    # loss: 1.0939 - accuracy: 0.9494 (no pre trained) 100 128 64 6
    # loss: 1.1126 - accuracy: 0.9310 (100d) 100 128 64 6

    # try with multiple output? label & subjectivity


def main():
    train3()


main()

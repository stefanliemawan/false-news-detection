import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from tokenization import normalize, process1, process2, returnStatementTokenizer

# Bi LSTM, Scikit Learn, Keras
# Mixed data neural network
# http: // digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/

vocab_size = 13304  # use maxlen?
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
    num_epochs = 100
    history = model.fit(
        x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    # 1 - 100 epoch - loss: 1.1556 - accuracy: 0.8879 (without pre trained glove)
    # 2 - 100 epoch - loss: 1.2017 - accuracy: 0.8416 (glove.6B.100d)


def createModel2(xseq_shape, xint_shape, n_output):
    seq_input = Input(shape=xseq_shape, name='seq_input')
    print(seq_input.shape)
    int_input = Input(shape=xint_shape, name='int_input')
    print(int_input.shape)
    emb = Embedding(vocab_size, embedding_dim,
                    input_length=xseq_shape[0])(seq_input)
    seq_out = Bidirectional(LSTM(embedding_dim, dropout=0.3,
                                 recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(seq_input)
    print(seq_out.shape)
    x = concatenate([seq_out, int_input])
    x = Dense(embedding_dim, activation='relu')(x)
    x = Dense(n_output, activation='softmax')(x)
    model = keras.Model(inputs=[seq_input, int_input], outputs=[x])
    return model


def train2():
    liar_train = normalize(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    # liar_train = shuffle(liar_train.reset_index(drop=True))
    # emb_matrix = glove(liar_train['statement'])
    xseq_train, xint_train, y_train = process2(liar_train)

    # print("emb_matrix = ", emb_matrix.shape)
    print('xseq_train shape = ', xseq_train.shape)
    print('xint_train shape = ', xint_train.shape)
    print('y_train shape = ', y_train.shape)

    n_output = max(y_train)+1

    model = createModel2(xseq_train.shape, xint_train.shape, n_output)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 5
    history = model.fit(
        np.array([xseq_train, xint_train]), y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    # ValueError: Failed to convert a NumPy array to a Tensor(Unsupported object type numpy.ndarray).


def main():
    train2()


main()

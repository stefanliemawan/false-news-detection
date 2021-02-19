import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils import np_utils
from sklearn.utils import shuffle
from tokenization import normalize, process, returnStatementTokenizer
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Bi LSTM, Scikit Learn, Keras
# Mixed data neural network
# http: // digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/

vocab_size = 15000  # use maxlen?
# vocab_size = 10375  # use maxlen?
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


def createModel(x_length1, x_length2, n_output1, n_output2):
    # add glove later to embedding
    x1 = Input(shape=(x_length1,), name="input_1")
    emb1 = Embedding(vocab_size, embedding_dim, name="embedding_1")(x1)
    cnn1 = Conv1D(128, 5, activation="relu", name="conv1d_1")(emb1)
    mp = GlobalMaxPooling1D(name="global_max_pooling1d_1")(cnn1)

    x2 = Input(shape=(x_length2,), name="input_2")
    emb2 = Embedding(vocab_size, embedding_dim, name="embedding_2")(x2)
    cnn2 = Conv1D(128, 5, activation="relu", name="conv1d_2")(emb2)
    bi_lstm = Bidirectional(LSTM(128, dropout=0.3,
                                 recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01), name="bidirectional_2"))(cnn2)
    bn = BatchNormalization(name="batch_normalization_2")(bi_lstm)

    x = concatenate([mp, bn])

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    # change subjectivity into regression? y2
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def train():
    liar_train = normalize(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    liar_train = shuffle(liar_train.reset_index(drop=True))
    liar_test = normalize(pd.read_csv('./cleanDatasets/clean_liar_test.csv'))
    liar_test = shuffle(liar_test.reset_index(drop=True))
    liar_valid = normalize(pd.read_csv('./cleanDatasets/clean_liar_valid.csv'))
    liar_valid = shuffle(liar_valid.reset_index(drop=True))

    # emb_matrix = glove(liar_train['statement'])

    print('Processing Training Data...')
    x_train1, x_train2, y_train1, y_train2 = process(liar_train)

    print('Processing Test Data...')
    x_test1, x_test2, y_test1, y_test2 = process(liar_test)

    print('Processing Validation Data...')
    x_val1, x_val2, y_val1, y_val2 = process(liar_valid)

    x_train1 = np.asarray(x_train1, dtype=np.float)
    x_train2 = np.asarray(x_train2, dtype=np.float)
    x_test1 = np.asarray(x_test1, dtype=np.float)
    x_test2 = np.asarray(x_test2, dtype=np.float)
    x_val1 = np.asarray(x_val1, dtype=np.float)
    x_val2 = np.asarray(x_val2, dtype=np.float)

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

    n_output1 = max(y_train1)+1
    n_output2 = max(y_train2)+1

    model = createModel(
        x_train1.shape[1], x_train2.shape[1], n_output1, n_output2)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 50
    model.fit(
        [x_train1, x_train2], [y_train1, y_train2], epochs=num_epochs, validation_data=([x_test1, x_test2], [y_test1, y_test2]), batch_size=batch_size, verbose=1)

    model.evaluate([x_test1, x_test2], [y_test1, y_test2], verbose=1)
    model.evaluate([x_val1, x_val2], [y_val1, y_val2], verbose=1)

    # train - loss: 1.6719 - output_1_loss: 1.0548 - output_2_loss: 0.6124 - output_1_accuracy: 0.9895 - output_2_accuracy: 0.9383 - val_loss: 2.7829 - val_output_1_loss: 1.7961 - val_output_2_loss: 0.9835 - val_output_1_accuracy: 0.2292 - val_output_2_accuracy: 0.5671

    # test - loss: 2.7829 - output_1_loss: 1.7961 - output_2_loss: 0.9835 - output_1_accuracy: 0.2292 - output_2_accuracy: 0.5671

    # val - loss: 2.7840 - output_1_loss: 1.7925 - output_2_loss: 0.9882 - output_1_accuracy: 0.2439 - output_2_accuracy: 0.5587

    # prediction = model.predict(x_val)
    # print(prediction)


def main():
    train()


main()

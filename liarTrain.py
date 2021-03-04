import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import gensim
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
from gensim.models.word2vec import Word2Vec


from tokenization import handleNaN, returnStatementTokenizer

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


def glove(texts):
    tokenizer = returnStatementTokenizer()
    tokenizer.fit_on_texts(texts.values)
    word_index = tokenizer.word_index
    global vocab_size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size = ', vocab_size)
    emb_index = {}
    with open("./glove/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            emb_index[word] = vector
    print('Loaded %s word vectors.' % len(emb_index))

    emb_matrix = np.zeros((vocab_size, embedding_dim))

    hits = 0
    misses = 0
    for word, i in word_index.items():
        emb_vector = emb_index.get(word)
        if emb_vector is not None:
            # words not found in embedding index will be all-zeros.
            emb_matrix[i] = emb_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return emb_matrix


def plot(history):
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()


def createModel(x_length1, x_length2, n_output1, n_output2, emb_matrix):
    # 1 not learning enough, stable at 0.2
    # 2 not learning anything
    x1 = Input(shape=(x_length1,), name="input_1")
    emb1 = Embedding(vocab_size, embedding_dim,  embeddings_initializer=keras.initializers.Constant(
        emb_matrix), trainable=False, name="embedding_1")(x1)
    cnn1 = Conv1D(128, 3, activation="relu", name="conv1d_1")(emb1)
    res1 = GlobalMaxPooling1D(name="global_max_pooling1d_1")(cnn1)

    x2 = Input(shape=(x_length2,), name="input_2")
    res2 = Dense(64, activation="relu")(x2)

    x = concatenate([res1, res2])

    y1 = Dense(n_output1, activation='softmax', name="output_1")(x)
    y2 = Dense(n_output2, activation='softmax', name="output_2")(x)
    # change subjectivity into regression? y2
    model = keras.Model(inputs=[x1, x2], outputs=[y1, y2])
    return model


def testModel1(x_shape, n_output1, emb_matrix):
    # overfit
    # word2vec?
    x1 = Input(shape=(x_shape[1],), name="input_1")
    emb1 = Embedding(vocab_size, embedding_dim,  embeddings_initializer=keras.initializers.Constant(
        emb_matrix), name="embedding_1")(x1)
    cnn1 = Conv1D(128, 3,
                  activation="relu", name="conv1d_1")(emb1)
    drop1 = Dropout(0.5, name="dropout_1")(cnn1)
    bn1 = BatchNormalization(name="batch_normalization_1")(drop1)
    mp1 = MaxPooling1D(3, name="max_pooling1d_1")(bn1)

    flat1 = Flatten(name="flat_1")(mp1)
    dense1 = Dense(64,
                   activation="relu", name="dense_1")(flat1)
    drop2 = Dropout(0.5, name="dropout_2")(dense1)
    y1 = Dense(n_output1, activation='softmax', name="output_1")(drop2)
    model = keras.Model(inputs=x1, outputs=y1)
    return model


def word2vec(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts.values)
    word_index = tokenizer.word_index
    global vocab_size
    vocab_size = len(word_index) + 1
    print('Vocabulary Size = ', vocab_size)

    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #     './word2vec/GoogleNews-vectors-negative300.bin', limit=50000, binary=True)
    model = gensim.models.KeyedVectors.load_word2vec_format(
        './word2vec/lexvec.commoncrawl.300d.W.pos.neg3.vectors', limit=50000, binary=False)
    print(model.most_similar('university'))

    emb_matrix = np.zeros((vocab_size, embedding_dim))

    hits = 0
    misses = 0
    for word, i in word_index.items():
        try:
            emb_vector = model[word]
            emb_matrix[i] = emb_vector
            hits += 1
        except Exception as ex:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    # pd.DataFrame(emb_matrix).to_csv(
    #     './matrix/liar-GoogleNews-vectors-negative300.csv')
    pd.DataFrame(emb_matrix).to_csv(
        './matrix/liar-lexvec-commoncrawl300.csv')

    # Google 50k limit 5810 misses
    # Lexvec 50k limit 4050 misses

    return emb_matrix


def train():
    liar_train = handleNaN(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv'))
    liar_test = handleNaN(pd.read_csv('./cleanDatasets/clean_liar_test.csv'))
    liar_valid = handleNaN(pd.read_csv('./cleanDatasets/clean_liar_valid.csv'))

    liar = pd.concat([liar_train, liar_test, liar_valid])
    liar = shuffle(liar)

    emb_matrix = word2vec(liar['statement'])
    print(emb_matrix)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(liar['statement'])
    sequences = tokenizer.texts_to_sequences(liar['statement'])
    maxlen = max([len(x) for x in sequences])
    padSequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=maxlen)
    x1 = np.array(padSequences)

    # vectorizer = TfidfVectorizer(max_features=300)
    # vectorizer = vectorizer.fit(liar['statement'])
    # x1 = vectorizer.transform(liar['statement']).toarray()

    encoder = LabelEncoder()
    encoder.fit(liar['label'])
    y1 = tf.keras.utils.to_categorical(encoder.transform(liar['label']))

    n_output1 = y1.shape[1]
    # n_output2 = y2.shape[1]

    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x1, y1, test_size=0.1, random_state=42)
    x_train1, x_val1, y_train1, y_val1 = train_test_split(
        x_train1, y_train1, test_size=0.1, random_state=42)

    # x_train2, x_test2, y_train2, y_test2 = train_test_split(
    #     x2, y2, test_size=0.1)
    # x_train2, x_val2, y_train2, y_val2 = train_test_split(
    #     x_train2, y_train2, test_size=0.1)

    print('x_train1 shape =', x_train1.shape)
    # print('x_train2 shape =', x_train2.shape)
    print('y_train1 shape =', y_train1.shape)
    # print('y_train2 shape =', y_train2.shape)
    print('x_test1 shape =', x_test1.shape)
    # print('x_test2 shape =', x_test2.shape)
    print('y_test1 shape =', y_test1.shape)
    # print('y_test2 shape =', y_test2.shape)
    print('x_val1 shape =', x_val1.shape)
    # print('x_val2 shape =', x_val2.shape)
    print('y_val1 shape =', y_val1.shape)
    # print('y_val2 shape =', y_val2.shape)

    # # model = createModel(
    # #     x_train1.shape[1], x_train2.shape[1], n_output1, n_output2, emb_matrix)
    model = testModel1(x_train1.shape, n_output1, emb_matrix)
    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer, metrics=['accuracy'])
    num_epochs = 12

    history = model.fit(x_train1, y_train1, epochs=num_epochs, validation_data=(
        x_val1, y_val1), batch_size=batch_size, shuffle=True, verbose=1)
    # plot(history)

    # model.fit(
    #     [x_train1, x_train2], [y_train1, y_train2], epochs=num_epochs, validation_data=([x_val1, x_val2], [y_val1, y_val2]), batch_size=batch_size, verbose=1)

    # kf = KFold(n_splits=3, shuffle=True)
    # k_fold = 1

    # for train_index, test_index in kf.split(x_train1):
    #     print("k = ", k_fold)

    #     k_x_train1, k_x_test1 = x_train1[train_index], x_train1[test_index]
    #     # k_x_train2, k_x_test2 = x_train2[train_index], x_train2[test_index]
    #     k_y_train1, k_y_test1 = y_train1[train_index], y_train1[test_index]
    #     # k_y_train2, k_y_test2 = y_train2[train_index], y_train2[test_index]
    #     model.fit(
    #         k_x_train1, k_y_train1, epochs=num_epochs, validation_data=(k_x_test1, k_y_test1), batch_size=batch_size, verbose=1)
    #     model.evaluate(k_x_test1,
    #                    k_y_test1, verbose=1)
    #     # model.fit(
    #     #     [k_x_train1, k_x_train2], [k_y_train1, k_y_train2], epochs=num_epochs, validation_data=([k_x_test1, k_x_test2], [k_y_test1, k_y_test2]), batch_size=batch_size, verbose=1)
    #     # model.evaluate([k_x_test1, k_x_test2], [
    #     #                k_y_test1, k_y_test2], verbose=1)

    #     k_fold += 1

    model.evaluate(x_val1, y_val1, verbose=1)
    model.evaluate(x_test1, y_test1, verbose=1)

    # model.evaluate([x_test1, x_test2], [y_test1, y_test2], verbose=1)
    # model.evaluate([x_val1, x_val2], [y_val1, y_val2], verbose=1)


def main():
    train()


main()

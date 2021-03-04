import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

vocab_size = 15000
embedding_dim = 100


def createModel():
    model = Sequential(n_output)
    model.add(Embedding(vocab_size, embedding_dim, name="embedding_1"))
    model.add(Conv1D(128, 5, activation="relu", name="conv1d_1"))
    model.add(GlobalMaxPooling1D(name="global_max_pooling1d_1"))
    model.add(Dense(n_output, activation='softmax', name="output_1"))
    return model


def main():
  # try liarTrain methods with faketrue dataset
    ft = handleNaN(pd.read_csv('./cleanDatasets/clean_fake_true.csv'))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(ft['text'])
    sequences = tokenizer.texts_to_sequences(ft['text'])
    wordIndex = tokenizer.word_index
    print('Vocabulary size: ', len(wordIndex))
    maxlen = max([len(x) for x in sequences])
    x = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=maxlen)

    encoder = LabelEncoder()
    encoder.fit(ft['label'].astype(str))
    y = encoder.transform(ft['label'])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1)


main()

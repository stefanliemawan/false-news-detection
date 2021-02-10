import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# Tokenize
# https://medium.com/@sthacruz/fake-news-classification-using-glove-and-long-short-term-memory-lstm-a48f1dd605ab

statement_tokenizer = Tokenizer()
subject_tokenizer = Tokenizer()
label_encoder = LabelEncoder()
speaker_encoder = LabelEncoder()
context_encoder = LabelEncoder()
sjt_encoder = LabelEncoder()
state_encoder = LabelEncoder()
party_encoder = LabelEncoder()
subjectivity_encoder = LabelEncoder()


def tokenize(tokenizer, data):
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    # wordIndex = tokenizer.word_index
    # print('Vocabulary size: ', len(wordIndex))
    maxlen = max([len(x) for x in sequences])
    padSequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=maxlen)
    # print('Shape of data tensor: ', padSequences.shape)
    return padSequences


def encode(encoder, data):
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    # dummy_y = np_utils.to_categorical(encoded_y)
    return encoded_y


def normalize(df):
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def process(data):
    label = encode(label_encoder, data['label'])
    statement = tokenize(
        statement_tokenizer, data['statement'])
    subject = tokenize(subject_tokenizer, data['subject'])
    speaker = encode(speaker_encoder, data['speaker'])
    sjt = encode(sjt_encoder, data["speaker's job title"])
    state = encode(state_encoder, data['state'])
    party = encode(party_encoder, data['party'])
    btc = np.array(data['barely true counts'])
    fc = np.array(data['false counts'])
    htc = np.array(data['half true counts'])
    mtc = np.array(data['mostly true counts'])
    potc = np.array(data['pants on fire counts'])
    context = encode(context_encoder, data['context'])
    polarity = np.array(data['polarity'])
    swc = np.array(data['subjectiveWordsCount'])
    subjectivity = encode(subjectivity_encoder, data['subjectivity'])

    x_train = list(map(list, zip(statement, subject, speaker,
                                 sjt, state, party, btc, fc, htc, mtc, potc, context, polarity, swc)))

    y_train = list(map(list, zip(label, subjectivity)))

    return np.array(x_train, dtype=object), np.array(y_train, dtype=object)

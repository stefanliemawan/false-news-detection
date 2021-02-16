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

maxl = 0


def returnStatementTokenizer():
    return statement_tokenizer


def tokenize(tokenizer, data):
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    # wordIndex = tokenizer.word_index
    # print('Vocabulary size: ', len(wordIndex))
    maxlen = max([len(x) for x in sequences])
    global maxl
    maxl = maxlen
    padSequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=maxlen)
    # print('Shape of data tensor: ', padSequences.shape)
    return padSequences


def encode(encoder, data):
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    # for i in range(len(encoded_y)):
    #     encoded_y[i] = pad(encoded_y[i])
    #     break
    # dummy_y = np_utils.to_categorical(encoded_y)
    return encoded_y


def pad(row):
    print(row)
    if len([row]) < maxl:
        row = np.array([row])
        row = np.pad(row, (0, maxl-len(row)), 'constant')
    print(row)
    return row


def normalize(df):
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def process1(data):
    statement = tokenize(
        statement_tokenizer, data['statement'])
    label = encode(label_encoder, data['label'])

    return statement, label


def process2(data):
    statement = tokenize(
        statement_tokenizer, data['statement'])
    label = encode(label_encoder, data['label'])
    subject = tokenize(subject_tokenizer, data['subject'])
    speaker = encode(speaker_encoder, data['speaker'])
    sjt = encode(sjt_encoder, data["speaker's job title"])
    state = encode(state_encoder, data['state'])
    party = encode(party_encoder, data['party'])
    btc = np.array(data['barely true counts'])
    fc = data['false counts']
    htc = data['half true counts']
    mtc = data['mostly true counts']
    potc = data['pants on fire counts']
    context = encode(context_encoder, data['context'])
    polarity = data['polarity']
    swc = data['subjectiveWordsCount']
    subjectivity = encode(subjectivity_encoder, data['subjectivity'])

    # xseq_train = list(map(list, zip(statement, subject)))
    # xseq_train = np.array(xseq_train, dtype=object)
    xseq_train = np.array(statement)

    xnum_train = list(map(list, zip(speaker,
                                    sjt, state, party, btc, fc, htc, mtc, potc, context, polarity, swc)))
    xnum_train = np.array(xnum_train, dtype=object)

    # y_train = list(map(list, zip(label, subjectivity)))
    # y_train = np.array(y_train, dtype=object)
    y_train = np.array(label)

    return xseq_train, xnum_train, y_train


def process3(data):
    statement = tokenize(
        statement_tokenizer, data['statement'])
    label = encode(label_encoder, data['label'])
    subject = tokenize(subject_tokenizer, data['subject'])
    speaker = encode(speaker_encoder, data['speaker'])
    sjt = encode(sjt_encoder, data["speaker's job title"])
    state = encode(state_encoder, data['state'])
    party = encode(party_encoder, data['party'])
    btc = np.array(data['barely true counts'])
    fc = data['false counts']
    htc = data['half true counts']
    mtc = data['mostly true counts']
    potc = data['pants on fire counts']
    context = encode(context_encoder, data['context'])
    polarity = data['polarity']  # minus value
    swc = data['subjectiveWordsCount']
    subjectivity = encode(subjectivity_encoder, data['subjectivity'])

    # xseq_train = list(map(list, zip(statement, subject)))
    # xseq_train = np.array(xseq_train, dtype=object)

    statement = np.array(statement)
    subject = np.array(subject)
    x_train = list(map(list, zip(speaker,
                                 sjt, state, party, btc, fc, htc, mtc, potc, context, swc)))
    x_train = np.array(x_train, dtype=object)

    x_train = np.concatenate((statement, subject, x_train), axis=1)

    # y_train = list(map(list, zip(label, subjectivity)))
    # y_train = np.array(y_train, dtype=object)
    y_train = np.array(label)

    return x_train, y_train

# tidy up process function later

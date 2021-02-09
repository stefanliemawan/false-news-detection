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


def returnStatementTokenizer():
    return statement_tokenizer


def returnLabelEncoder():
    return label_encoder


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
    data = data.drop('id', axis=1)
    data['statement'] = tokenize(statement_tokenizer, data['statement'])
    data['subject'] = tokenize(subject_tokenizer, data['subject'])
    data['speaker'] = encode(speaker_encoder, data['speaker'])
    data['context'] = encode(context_encoder, data['context'])
    data["speaker's job title"] = encode(sjt_encoder,
                                         data["speaker's job title"])
    data['state'] = encode(state_encoder, data['state'])
    data['party'] = encode(party_encoder, data['party'])

    data['label'] = encode(label_encoder, data['label'])
    data['subjectivity'] = encode(subjectivity_encoder, data['subjectivity'])
    return data


def main():
    liar_train = process(normalize(pd.read_csv(
        './cleanDatasets/clean_liar_train.csv')))
    print(liar_train)
    liar_train.to_pickle('./cleanDatasets/tokenized_liar_train.pkl')
    liar_test = process(normalize(pd.read_csv(
        './cleanDatasets/clean_liar_test.csv')))
    liar_test.to_pickle('./cleanDatasets/tokenized_liar_test.pkl')
    liar_valid = process(normalize(pd.read_csv(
        './cleanDatasets/clean_liar_valid.csv')))
    liar_valid.to_pickle('./cleanDatasets/tokenized_liar_valid.pkl')


main()

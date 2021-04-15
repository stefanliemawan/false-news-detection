import pandas as pd
import numpy as np
import tensorflow as tf
import gensim
import calendar
import os
import sys
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec

# Tokenize
# https://medium.com/@sthacruz/fake-news-classification-using-glove-and-long-short-term-memory-lstm-a48f1dd605ab

statement_tokenizer = Tokenizer(num_words=12000)
subject_tokenizer = Tokenizer()

context_encoder = LabelEncoder()
speaker_encoder = LabelEncoder()
sjt_encoder = LabelEncoder()
state_encoder = LabelEncoder()
party_encoder = LabelEncoder()
checker_encoder = LabelEncoder()

label_encoder = LabelEncoder()
subj_encoder = LabelEncoder()

vocab_size = 0


def returnVocabSize():
    return vocab_size


def tokenizeStatement(tokenizer, data):
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    # print(word_index)
    global vocab_size
    vocab_size = len(word_index) + 1
    # print(word_index)
    print('Statement Vocabulary size', vocab_size)
    # wi = [i for i, j in word_index.items() if len(i) == 1]
    # print(wi[:100])
    # np.savetxt('statement_word_index.cxv', word_index, delimiter=",")
    maxlen = max([len(x) for x in sequences])
    # print(np.mean([len(x) for x in sequences]))
    # maxlen = 50
    # play with maxlen?

    padded_sequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=maxlen)
    # print('Shape of data tensor: ', padded_sequences.shape)
    return np.array(padded_sequences)


def tokenizeSubject(tokenizer, data):
    data = [x.split(',') for x in data]
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    # print('Vocabulary size: ', len(word_index))
    maxlen = max([len(x) for x in sequences])
    # maxlen = 20

    padded_sequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=5)
    # print('Shape of data tensor: ', padded_sequences.shape)
    return np.array(padded_sequences)


def encode(encoder, data, onehot):
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    if onehot == True:
        dummy_y = tf.keras.utils.to_categorical(encoded_y)
        return np.array(dummy_y)
    else:
        return np.array(encoded_y)


def gloveMatrix(statements):
    global statement_tokenizer
    statement_tokenizer.fit_on_texts(statements)
    word_index = statement_tokenizer.word_index
    vocab_size = len(statement_tokenizer.word_index) + 1
    print('Vocabulary Size = ', vocab_size)

    emb_index = {}
    glove_path = "./glove/glove.6B.300d.txt"
    # glove_path = "./glove/glove.42B.300d.txt"
    # glove_path = "./glove/glove.twitter.27B.200d.txt"
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            emb_index[word] = vector
    print('Loaded %s word vectors.' % len(emb_index))

    emb_matrix = np.zeros((vocab_size, 300))

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
    print("Glove Converted %d words (%d misses)" % (hits, misses))

    return emb_matrix


def word2vecMatrix(statements):
    global statement_tokenizer
    statement_tokenizer.fit_on_texts(statements)
    word_index = statement_tokenizer.word_index
    vocab_size = len(statement_tokenizer.word_index) + 1
    print('Word2Vec Vocabulary Size', vocab_size)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        './word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    # w2v=gensim.models.KeyedVectors.load_word2vec_format(
    #     './word2vec/GoogleNews-vectors-negative300.bin', limit=50000, binary=True)
    # limit max around 1m

    sentences = [sentence.split() for sentence in statements]
    maxlen = max([len(x) for x in sentences])
    x1 = []

    emb_matrix = np.zeros((vocab_size, 300))

    hits = 0
    misses = 0
    for word, i in word_index.items():
        try:
            emb_matrix[i] = w2v[word]
            hits += 1
        except:
            misses += 1
    print("Word2vec Converted %d words (%d misses)" % (hits, misses))

    return emb_matrix


def word2vecInput(statements):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        './word2vec/GoogleNews-vectors-negative300.bin', limit=1000000, binary=True)
    # 1m - 12683 misses
    # 3m words

    sentences = [sentence.split() for sentence in statements]
    maxlen = max([len(x) for x in sentences])
    wnf = [0 for x in range(300)]
    x1 = []

    hits = 0
    misses = 0
    for sentence in sentences:
        s = []
        for word in sentence:
            try:
                vector = w2v[word]
                hits += 1
            except:
                vector = wnf
                misses += 1
            s.append(vector)
        for i in range(maxlen-len(s)):
            s.append(wnf)
        x1.append(s)

    print("Word2vec Converted %d words (%d misses)" % (hits, misses))
    x1 = np.array(x1, dtype=float)
    return x1


def processPoliti(data):
    statement = tokenizeStatement(
        statement_tokenizer, data['statement'],)
    speaker = encode(speaker_encoder, data['speaker'], False)
    # swc = data['subjectiveWordsCount']
    polarity = np.array(data['polarity'])
    label = encode(label_encoder, data['label'], True)
    subjectivity = encode(subj_encoder, data['subjectivity'], True)

    tags = np.array(
        data.drop(['label', 'statement', 'speaker', 'subjectivity'], axis=1))

    # label_dict = dict(zip(label_encoder.classes_,
    #                       label_encoder.transform(label_encoder.classes_)))
    # label_dict = {str(v): k for k, v in label_dict.items()}
    # subj_dict = dict(zip(subj_encoder.classes_,
    #                      subj_encoder.transform(subj_encoder.classes_)))
    # subj_dict = {str(v): k for k, v in subj_dict.items()}

    # print(label_dict)
    # print(subj_dict)

    # with open('label_mapping.json', 'w') as f:
    #     json.dump(label_dict, f)
    # with open('subj_mapping.json', 'w') as f:
    #     json.dump(subj_dict, f)

    x1 = statement
    x2 = np.reshape(speaker, (-1, 1))  # 0.36
    print('Speaker Length', max(speaker))
    # x2 = np.column_stack((speaker, polarity))
    # x2 = np.column_stack((speaker, polarity, tags)) # 0.35
    # x2 = tags # 0.34
    # try with these three

    y1 = label
    y2 = subjectivity

    return x1, x2, y1, y2

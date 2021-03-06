import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.models.word2vec import Word2Vec

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

statement_maxl = 0
subject_maxl = 0


def returnStatementTokenizer():
    global statement_tokenizer
    return statement_tokenizer


def tokenize(tokenizer, data, col):
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    wordIndex = tokenizer.word_index
    # print('Vocabulary size: ', len(wordIndex))
    maxlen = max([len(x) for x in sequences])

    # to make sure x_train and y_train have the same columns, split function later
    if col == 'statement':
        global statement_maxl
        if statement_maxl == 0:
            statement_maxl = maxlen
        else:
            maxlen = statement_maxl
    elif col == 'subject':
        global subject_maxl
        if subject_maxl == 0:
            subject_maxl = maxlen
        else:
            maxlen = subject_maxl

    padSequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=maxlen)
    # print('Shape of data tensor: ', padSequences.shape)
    return padSequences


def encode(encoder, data, onehot):
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    if onehot == True:
        dummy_y = tf.keras.utils.to_categorical(encoded_y)
        return dummy_y
    else:
        return encoded_y


def tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=300)
    vectorizer = vectorizer.fit(texts)
    tfidf = vectorizer.transform(texts).toarray()
    return tfidf


def handleNaN(df):
    # find a way not to drop rows with nan
    for header in df.columns.values:
        df[header] = df[header].fillna(
            df[header][df[header].first_valid_index()])
    # df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    return df


def glove(texts):
    global statement_tokenizer
    tokenizer = statement_tokenizer
    tokenizer.fit_on_texts(texts.values)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size = ', vocab_size)

    emb_index = {}
    with open("./glove/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
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


def word2vec(statements):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        './word2vec/GoogleNews-vectors-negative300.bin', limit=1000000, binary=True)

    sentences = [sentence.split() for sentence in statements]
    maxlen = max([len(x) for x in sentences])
    x1 = []

    hits = 0
    misses = 0
    wnf = [0 for x in range(300)]
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


def process(data):
    global statement_maxl, subject_maxl

    statement = tokenize(
        statement_tokenizer, data['statement'], 'statement')
    label = encode(label_encoder, data['label'], True)
    # subject = tokenize(subject_tokenizer, data['subject'], 'subject')
    speaker = encode(speaker_encoder, data['speaker'], False)
    sjt = encode(sjt_encoder, data["speaker's job title"], False)
    state = encode(state_encoder, data['state'], False)
    party = encode(party_encoder, data['party'], False)
    # btc = data['barely true counts']
    # fc = data['false counts']
    # htc = data['half true counts']
    # mtc = data['mostly true counts']
    # potc = data['pants on fire counts']
    # context = encode(context_encoder, data['context'], False)
    # polarity = data['polarity']  # minus value
    swc = data['subjectiveWordsCount']
    subjectivity = encode(subjectivity_encoder, data['subjectivity'], True)

    x1 = word2vec(data['statement'])
    # x1 = np.asarray(statement)

    x2 = list(map(list, zip(speaker, sjt, state, party, swc)))
    x2 = np.asarray(x2)

    y1 = np.asarray(label)
    y2 = np.asarray(subjectivity)

    return x1, x2, y1, y2

# tidy up process function later

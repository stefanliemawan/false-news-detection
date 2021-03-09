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

context_encoder = LabelEncoder()
speaker_encoder = LabelEncoder()
sjt_encoder = LabelEncoder()
state_encoder = LabelEncoder()
party_encoder = LabelEncoder()

label_encoder = LabelEncoder()
subjectivity_encoder = LabelEncoder()


def tokenizeStatement(tokenizer, data):
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    wordIndex = tokenizer.word_index
    # print('Vocabulary size: ', len(wordIndex))
    maxlen = max([len(x) for x in sequences])
    # maxlen = 47

    padSequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=25)
    # print('Shape of data tensor: ', padSequences.shape)
    return np.array(padSequences)


def tokenizeSubject(tokenizer, data):
    data = [x.split(',') for x in data]
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    wordIndex = tokenizer.word_index
    # print('Vocabulary size: ', len(wordIndex))
    maxlen = max([len(x) for x in sequences])
    # maxlen = 20

    padSequences = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=5)
    # print('Shape of data tensor: ', padSequences.shape)
    return np.array(padSequences)


def handleContext(data):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(data)
    df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    context = df.idxmax(axis=1)
    encoded_context = encode(context_encoder, data, False)
    return np.array(encoded_context)


def encode(encoder, data, onehot):
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    if onehot == True:
        dummy_y = tf.keras.utils.to_categorical(encoded_y)
        return np.array(dummy_y)
    else:
        return np.array(encoded_y)


def tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=300)
    vectorizer = vectorizer.fit(texts)
    tfidf = vectorizer.transform(texts).toarray()
    return tfidf


def gloveMatrix(statements):
    global statement_tokenizer
    statement_tokenizer.fit_on_texts(statements)
    word_index = statement_tokenizer.word_index
    vocab_size = len(statement_tokenizer.word_index) + 1
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


def word2vecMatrix(statements):
    global statement_tokenizer
    statement_tokenizer.fit_on_texts(statements)
    word_index = statement_tokenizer.word_index
    vocab_size = len(statement_tokenizer.word_index) + 1
    print('Vocabulary Size = ', vocab_size)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        './word2vec/GoogleNews-vectors-negative300.bin', limit=100000, binary=True)
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


def process(data):
    statement = tokenizeStatement(
        statement_tokenizer, data['statement'],)
    subject = tokenizeSubject(subject_tokenizer, data['subject'])

    speaker = encode(speaker_encoder, data['speaker'], False)
    sjt = encode(sjt_encoder, data["speaker's job title"], False)
    state = encode(state_encoder, data['state'], False)
    party = encode(party_encoder, data['party'], False)
    context = handleContext(data['context'])
    swc = data['subjectiveWordsCount']
    # btc = data['barely true counts']
    # fc = data['false counts']
    # htc = data['half true counts']
    # mtc = data['mostly true counts']
    # potc = data['pants on fire counts']
    # polarity = data['polarity']

    label = encode(label_encoder, data['label'], True)
    subjectivity = encode(subjectivity_encoder, data['subjectivity'], True)

    # x1 = word2vecInput(data['statement'])
    # x1 = tfidf(data['statement'])

    # x1 = []
    # for i in range(len(statement)):
    #     x1.append(np.concatenate((statement[i], subject[i])))

    # x1 = np.asarray(x1)
    # print(x1[0:10])

    x1 = statement

    x2 = np.array(
        list(map(list, zip(speaker, sjt, state, party, context))))
    x2 = np.concatenate((subject, x2), axis=1)

    y1 = label
    y2 = subjectivity

    return x1, x2, y1, y2

# tidy up process function later

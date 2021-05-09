from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bert import tokenization
from tokenizers import BertWordPieceTokenizer
from transformers import DistilBertTokenizer, DistilBertConfig
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os
import calendar
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
tf.gfile = tf.io.gfile

# Define encoders
context_encoder = LabelEncoder()
speaker_encoder = LabelEncoder()
sjt_encoder = LabelEncoder()
state_encoder = LabelEncoder()
party_encoder = LabelEncoder()
context_encoder = LabelEncoder()

label_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()
subjectivity_encoder = LabelEncoder()

subject_tokenizer = Tokenizer()

st_maxlen = 0
md_maxlen = 0
vocab_size = 0

# Define DistilBERT
dbert_tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased')


def tokenizeBert(data, statement=True):  # Tokenization for DistilBERT
    global dbert_tokenizer
    global st_maxlen
    global md_maxlen

    if statement == True:
        sequence = [x.split() for x in data]
        maxlen = max([len(x) for x in sequence]) + 2
        st_maxlen = max(st_maxlen, maxlen)
        maxlen = st_maxlen
    else:
        data["subject"] = data["subject"].str.replace(",", " ")
        if "date" in data.columns:
            data["date"] = data["date"].str.replace(",", " ")

        data = [" ".join(x) for x in data.values]
        sequence = [x.split() for x in data]

        maxlen = max([len(x) for x in sequence])+2
        md_maxlen = max(md_maxlen, maxlen)
        maxlen = md_maxlen

    input_ids = []
    attention_mask = []

    for text in data:
        encoding = dbert_tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=maxlen, padding="max_length", truncation=True)

        input_ids.append(encoding["input_ids"])
        attention_mask.append(encoding["attention_mask"])

    return np.asarray(input_ids).astype(int), np.asarray(attention_mask).astype(int)


def tokenizeSubject(data):
    global subject_tokenizer
    data = [x.split(",") for x in data]
    subject_tokenizer.fit_on_texts(data)
    sequences = subject_tokenizer.texts_to_sequences(data)
    word_index = subject_tokenizer.word_index
    # print("Vocabulary size: ", len(word_index))
    maxlen = max([len(x) for x in sequences])
    # print("Subject", np.mean([len(x) for x in sequences]))
    # maxlen = 20

    padded_sequences = pad_sequences(
        sequences, padding="post", truncating="post", maxlen=5)
    # print("Shape of data tensor: ", padded_sequences.shape)
    return np.array(padded_sequences)


def encode(encoder, data, onehot):  # Encode function
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    if onehot == True:
        dummy_y = tf.keras.utils.to_categorical(encoded_y)
        return np.array(dummy_y)
    else:
        return np.array(encoded_y)


def handleDate(data):  # Handle date column
    date = [d.split("-") for d in data]
    res = []
    m = {month: index for index, month in enumerate(
        calendar.month_abbr) if month}
    for d in date:
        if len(d) < 3:
            day, month, year = 0, 0, 0
        else:
            day = int(d[0])
            month = int(m[d[1]])
            year = int(str(20)+str(d[2]))
        res.append([day, month, year])
    return np.array(res)


def gloveMatrix(statements):  # Create GloVE embedding matrix
    word_index = statement_tokenizer.word_index

    emb_index = {}
    glove_path = "./glove/glove.6B.300d.txt"
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            emb_index[word] = vector
    print("Loaded %s word vectors." % len(emb_index))

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
    print("GloVe Converted %d words (%d misses)" % (hits, misses))

    return emb_matrix


def word2vecMatrix(statements):  # Create Word2Vec embedding matrix
    word_index = statement_tokenizer.word_index
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        "./word2vec/GoogleNews-vectors-negative300.bin", limit=50000, binary=True)

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


def fasttextMatrix(statements):  # Create FastText embedding matrix
    word_index = statement_tokenizer.word_index
    ft = gensim.models.KeyedVectors.load_word2vec_format(
        "./fasttext/crawl-300d-2M.vec")

    sentences = [sentence.split() for sentence in statements]
    maxlen = max([len(x) for x in sentences])
    x1 = []

    emb_matrix = np.zeros((vocab_size, 300))

    hits = 0
    misses = 0
    for word, i in word_index.items():
        try:
            emb_matrix[i] = ft[word]
            hits += 1
        except:
            misses += 1
    print("FastText Converted %d words (%d misses)" % (hits, misses))

    return emb_matrix


def process(data):  # Main function to handle all preprocessing of different columns
    input_ids1, attention_mask1 = tokenizeBert(
        data["statement"], True)
    if "date" in data.columns:
        input_ids2, attention_mask2 = tokenizeBert(
            data[["speaker", "speaker's job title", "state", "party", "date", "context", "subject"]], False)
    else:
        input_ids2, attention_mask2 = tokenizeBert(
            data[["speaker", "speaker's job title", "state", "party", "context", "subject"]], False)

    score = np.array(data["credit score"]).astype(int)

    if "true counts" in data.columns:
        t_counts = np.array(data["true counts"]).astype(int)
    mt_counts = np.array(data["mostly true counts"]).astype(int)
    ht_counts = np.array(data["half true counts"]).astype(int)
    mf_counts = np.array(data["mostly false counts"]).astype(int)
    f_counts = np.array(data["false counts"]).astype(int)
    pf_counts = np.array(data["pants on fire counts"]).astype(int)
    polarity = np.array(data["polarity"]).astype(int)

    label = encode(label_encoder, data["label"], True)
    subjectivity = encode(subjectivity_encoder, data["subjectivity"], True)

    x1 = {"input_ids": input_ids1,
          "attention_mask": attention_mask1}
    x2 = {"input_ids": input_ids2,
          "attention_mask": attention_mask2}
    x3 = np.reshape(score, (-1, 1))

    if "true counts" in data.columns:
        x4 = np.column_stack((t_counts, mt_counts,
                              ht_counts, mf_counts, f_counts, pf_counts, polarity))
    else:
        x4 = np.column_stack((mt_counts,
                              ht_counts, mf_counts, f_counts, pf_counts, polarity))

    y1 = label
    y2 = subjectivity

    return [x1, x2, x3, x4], [y1, y2]

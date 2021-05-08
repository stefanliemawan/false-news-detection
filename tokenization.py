from bert import tokenization
import tensorflow_hub as hub
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, DistilBertTokenizer, DistilBertConfig
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import sys
import os
import calendar
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
tf.gfile = tf.io.gfile

statement_tokenizer = Tokenizer(num_words=None)
subject_tokenizer = Tokenizer()

context_encoder = LabelEncoder()
speaker_encoder = LabelEncoder()
sjt_encoder = LabelEncoder()
state_encoder = LabelEncoder()
party_encoder = LabelEncoder()
context_encoder = LabelEncoder()

label_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()
subjectivity_encoder = LabelEncoder()

statement_maxlen = 0
metadata_maxlen = 8
vocab_size = 0

# bert_tiny = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"
# bert_miny = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2"
# bert_small = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
# bert_medium = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
# bert_base = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

# bert_layer = hub.KerasLayer(bert_medium, trainable=True)
# vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# bert_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

dbert_tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased')


def returnBertLayer():
    return bert_layer


def returnVocabSize():
    return vocab_size


def returnStatementMaxlen():
    return statement_maxlen


def returnMetadataMaxlen():
    return metadata_maxlen


def tokenizeBert(data, statement=True):
    global dbert_tokenizer
    global statement_maxlen
    global metadata_maxlen

    maxlen = 45
    # sequence = [x.split() for x in data]
    # maxlen = max([len(x) for x in sequence]) + 2
    # statement_maxlen = max(statement_maxlen, maxlen)
    # maxlen = statement_maxlen
    if statement == False:
        data = [" ".join(x) for x in data.values]
        # applied as a sentence

    input_ids = []
    attention_mask = []

    for text in data:
        encoding = dbert_tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=maxlen, padding="max_length", truncation=True)

        input_ids.append(encoding["input_ids"])
        attention_mask.append(encoding["attention_mask"])

    return np.asarray(input_ids).astype(int), np.asarray(attention_mask).astype(int)


def tokenizeStatement(data):
    global statement_tokenizer
    statement_tokenizer.fit_on_texts(data)
    sequences = statement_tokenizer.texts_to_sequences(data)
    word_index = statement_tokenizer.word_index

    global vocab_size
    vocab_size = len(word_index) + 1
    print("Statement Vocabulary size", vocab_size)
    global statement_maxlen
    maxlen = max([len(x) for x in sequences])
    statement_maxlen = max(statement_maxlen, maxlen)
    # print(np.mean([len(x) for x in sequences]))
    # maxlen = 50

    padded_sequences = pad_sequences(
        sequences, padding="post", truncating="post", maxlen=statement_maxlen)
    # print("Shape of data tensor: ", padded_sequences.shape)
    return np.array(padded_sequences)


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
        sequences, padding="post", truncating="post", maxlen=maxlen)
    # print("Shape of data tensor: ", padded_sequences.shape)
    return np.array(padded_sequences)


def encode(encoder, data, onehot):
    encoder.fit(data.astype(str))
    encoded_y = encoder.transform(data)
    if onehot == True:
        dummy_y = tf.keras.utils.to_categorical(encoded_y)
        return np.array(dummy_y)
    else:
        return np.array(encoded_y)


def handleDate(data):
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


def gloveMatrix(statements):
    word_index = statement_tokenizer.word_index

    emb_index = {}
    glove_path = "./glove/glove.6B.300d.txt"
    # glove_path = "./glove/glove.42B.300d.txt"
    # glove_path = "./glove/glove.twitter.27B.200d.txt"
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


def word2vecMatrix(statements):
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


def fasttextMatrix(statements):
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


def bertMatrix(statements):
    pass
    # bert = BertModel.from_pretrained(BERT_FP)
    # bert_embeddings = list(bert.children())[0]
    # bert_word_embeddings = list(bert_embeddings.children())[0]
    # mat = bert_word_embeddings.weight.data.numpy()
    # return mat


def processLiar(data):
    input_ids1, attention_mask1 = tokenizeBert(
        data["statement"], True)
    input_ids2, attention_mask2 = tokenizeBert(
        data[["speaker", "speaker's job title", "state", "party", "context", "subject"]], False)

    # subject = tokenizeSubject(data["subject"])
    # context = encode(context_encoder, data["context"], False)
    # speaker = encode(speaker_encoder, data["speaker"], False)
    # sjt = encode(sjt_encoder, data["speaker's job title"], False)
    # state = encode(state_encoder, data["state"], False)
    # party = encode(party_encoder, data["party"], False)

    # t_counts = np.array(data["true counts"]).astype(int)
    mt_counts = np.array(data["mostly true counts"]).astype(int)
    ht_counts = np.array(data["half true counts"]).astype(int)
    mf_counts = np.array(data["mostly false counts"]).astype(int)
    f_counts = np.array(data["false counts"]).astype(int)
    pf_counts = np.array(data["pants on fire counts"]).astype(int)
    score = np.array(data["credit score"]).astype(int)

    polarity = np.array(data["polarity"]).astype(int)
    sentiment = encode(sentiment_encoder, data["sentiment"], False)
    tags = np.array(data.loc[:, "CC":"VBZ"]).astype(int)

    label = encode(label_encoder, data["label"], True)
    subjectivity = encode(subjectivity_encoder, data["subjectivity"], True)

    x1 = {"input_ids": input_ids1,
          "attention_mask": attention_mask1}
    x2 = {"input_ids": input_ids2,
          "attention_mask": attention_mask2}

    x3 = np.column_stack((mt_counts,
                          ht_counts, mf_counts, f_counts, pf_counts))
    x4 = np.column_stack((score, polarity, tags))
    # x4 = tags

    y1 = label
    y2 = subjectivity

    return x1, x2, x3, x4, y1, y2


def processPoliti(data):
    tokens1, masks1, segments1 = tokenizeBert(
        data["statement"], True)
    tokens2, masks2, segments2 = tokenizeBert(
        data[["speaker", "speaker's job title", "state", "party", "context", "subject"]], False)

    # subject = tokenizeSubject(data["subject"])
    # speaker = encode(speaker_encoder, data["speaker"], False)
    # sjt = encode(sjt_encoder, data["speaker's job title"], False)
    # state = encode(state_encoder, data["state"], False)
    # party = encode(party_encoder, data["party"], False)
    # date = handleDate(data['date'])
    # context = encode(context_encoder, data["context"], False)

    t_counts = np.array(data["true counts"]).astype(int)
    mt_counts = np.array(data["mostly true counts"]).astype(int)
    ht_counts = np.array(data["half true counts"]).astype(int)
    mf_counts = np.array(data["mostly false counts"]).astype(int)
    f_counts = np.array(data["false counts"]).astype(int)
    pf_counts = np.array(data["pants on fire counts"]).astype(int)
    score = np.array(data["credit score"]).astype(int)

    polarity = np.array(data["polarity"]).astype(int)
    sentiment = encode(sentiment_encoder, data["sentiment"], False)
    tags = np.array(data.loc[:, "CC":"VBZ"]).astype(int)

    label = encode(label_encoder, data["label"], True)
    subjectivity = encode(subjectivity_encoder, data["subjectivity"], True)

    x1 = {"input_word_ids": tokens1, "input_mask": masks1,
          "input_type_ids": segments1}
    x2 = {"input_word_ids": tokens2, "input_mask": masks2,
          "input_type_ids": segments2}
    # x2 = np.column_stack((subject, speaker, sjt, state, party, context, date))
    x3 = np.column_stack((mt_counts,
                          ht_counts, mf_counts, f_counts, pf_counts))
    x4 = np.column_stack((score, polarity, tags))

    y1 = label
    y2 = subjectivity

    return x1, x2, x3, x4, y1, y2

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import tokenize

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer

# token_space = tokenize.WhitespaceTokenizer()

# tfidf from
# https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
# glove from
# https://www.kaggle.com/hassanamin/fake-news-classifier-using-glove


def tfidf(textList):
    tfidfVectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf = tfidfVectorizer.fit_transform(textList)
    tfidf_tokens = tfidfVectorizer.get_feature_names()
    df_tfidfvect = pd.DataFrame(data=tfidf.toarray(), columns=tfidf_tokens)

    print("\nTF-IDF Vectorizer\n")
    print(df_tfidfvect)
    return df_tfidfvect


def count(textList):
    countVectorizer = CountVectorizer(analyzer='word', stop_words='english')
    count = countVectorizer.fit_transform(textList)
    count_tokens = countVectorizer.get_feature_names()
    df_countvect = pd.DataFrame(data=count.toarray(), columns=count_tokens)

    print("Count Vectorizer\n")
    print(df_countvect)
    return df_countvect


def glove(textList, embedding_dim):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(textList.values)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    emb_index = {}
    with open("./glove/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            emb_index[word] = vector
    print('Loaded %s word vectors.' % len(emb_index))

    # emb_matrix = np.zeros((vocab_size, embedding_dim))
    # for word, i in word_index.items():
    #     emb_vector = emb_index.get(word)
    #     if emb_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         emb_matrix[i] = emb_vector

    emb_matrix = pd.DataFrame(
        data=emb_index, columns=tokenizer.word_index.keys())
    print(emb_matrix)
    return emb_matrix


def main():
    # fakeTrueData = pd.read_csv('./cleanDatasets/clean_fake_true.csv')
    liarTrainData = pd.read_csv('./cleanDatasets/clean_liar_train.csv')
    liarTestData = pd.read_csv('./cleanDatasets/clean_liar_test.csv')
    liarValidData = pd.read_csv('./cleanDatasets/clean_liar_valid.csv')
    tf = tfidf(liarTrainData['statement'])
    # tf.to_csv('./matrix/tfidf_liar_train.csv')
    # gLTrainData = glove(liarTrainData['statement'])
    # gLTestData = glove(liarTestData['statement'])
    # gLValidData = glove(liarValidData['statement'])
    # gLTrainData.to_csv('./matrix/glove_liar_train.csv')
    # gLTestData.to_csv('./matrix/glove_liar_test.csv')
    # gLValidData.to_csv('./matrix/glove_liar_valid.csv')
    # glove(liarTestData['statement'])


main()

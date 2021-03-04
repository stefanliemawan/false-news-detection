import pandas as pd
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.utils import shuffle

from nltk.corpus import stopwords
from textblob import Word


def clean(df):
    df['statement'] = df['statement'].str.lower()
    # clean stop words?
    # stop = stopwords.words('english')
    # df['statement'] = df['statement'].apply(
    #     lambda x: ' '.join(x.lower() for x in x.split()))
    # df['statement'] = df['statement'].apply(lambda x: ' '.join(
    #     x for x in x.split() if x not in string.punctuation))
    # df['statement'] = df['statement'].str.replace('[^\w\s]', '')
    # df['statement'] = df['statement'].apply(
    #     lambda x: ' '.join(x for x in x.split() if not x.isdigit()))
    # df['statement'] = df['statement'].apply(
    #     lambda x: ' '.join(x for x in x.split() if not x in stop))
    # df['statement'] = df['statement'].apply(lambda x: " ".join(
    #     [Word(word).lemmatize() for word in x.split()]))
    return df


def main():
    liar_train = pd.read_csv(
        './datasets/LIAR/liar_train_labeled.csv')
    liar_test = pd.read_csv('./datasets/LIAR/liar_test_labeled.csv')
    liar_valid = pd.read_csv(
        './datasets/LIAR/liar_valid_labeled.csv')

    liar = clean(pd.concat([liar_train, liar_test, liar_valid]))
    liar = shuffle(liar)

    sentences = [sentence.split() for sentence in liar['statement']]
    model = Word2Vec(sentences)
    # model.save('./word2vec/liar_word2vec.bin')
    # model.train(sentences, total_examples=1, epochs=1)
    print(model.wv.most_similar('deficit'))
    vectors = []

    # for sentence in sentences:
    #     for word in sentence:


main()

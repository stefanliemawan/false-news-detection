import nltk
from nltk import tokenize
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def getSubjectiveWordsCount(row):
    taggedText = nltk.pos_tag(nltk.word_tokenize(row.statement))
    # taggedText = nltk.pos_tag(nltk.word_tokenize(row.text))
    adwordsCount = 0
    for tag in taggedText:
        if (tag[1][0] == 'J') or (tag[1][0] == 'R'):
            adwordsCount += 1
    return adwordsCount


def getSubjectivity(row):
    subjectivity = round(TextBlob(row.statement).sentiment.subjectivity, 2)
    # subjectivity = round(TextBlob(row.text).sentiment.subjectivity,1)
    # return subjectivity

    if (subjectivity == 0.00):
        return 'VERY-LOW'
    elif (subjectivity > 0.00) and (subjectivity <= 0.2):
        return 'LOW'
    elif (subjectivity > 0.2) and (subjectivity <= 0.4):
        return 'MEDIUM'
    elif (subjectivity > 0.6) and (subjectivity <= 0.8):
        return 'HIGH'
    elif (subjectivity > 0.8):
        return 'VERY-HIGH'

    # if (subjectivity <= 0.2):
    #     return 'LOW'
    # elif (subjectivity > 0.2) and (subjectivity <= 0.5):
    #     return 'MEDIUM'
    # elif (subjectivity > 0.5):
    #     return 'HIGH'

    # analyzer = SentimentIntensityAnalyzer()
    # vs = analyzer.polarity_scores(row.statement)
    # c = max(vs, key=vs.get)
    # return c

    # play around with the value


def getPolarity(row):
    polarity = TextBlob(row.statement).sentiment.polarity + 1
    # if polarity < 0:
    #     polarity = 0
    return polarity


def applyToDF(df):
    # df['polarity'] = df.apply(getPolarity, axis=1)
    # df['subjectiveWordsCount'] = df.apply(getSubjectiveWordsCount, axis=1)
    df['subjectivity'] = df.apply(
        getSubjectivity, axis=1)
    return df


def liar():
    liar_train_path = 'cleanDatasets/clean_liar_train.csv'
    liar_test_path = 'cleanDatasets/clean_liar_test.csv'
    liar_valid_path = 'cleanDatasets/clean_liar_valid.csv'

    liar_train = pd.read_csv(liar_train_path)
    liar_test = pd.read_csv(liar_test_path)
    liar_valid = pd.read_csv(liar_valid_path)

    # Can be put into a function with params data

    liar_train.to_csv(liar_train_path, encoding='utf-8-sig', index=False)
    liar_test.to_csv(liar_test_path, encoding='utf-8-sig', index=False)
    liar_valid.to_csv(liar_valid_path, encoding='utf-8-sig', index=False)


def politifact():
    path = 'cleanDatasets/clean_politifact.csv'
    data = pd.read_csv(path)
    data = applyToDF(data)
    data.to_csv(path, index=False)

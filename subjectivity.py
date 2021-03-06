import nltk
from nltk import tokenize
import pandas as pd
import numpy as np
from textblob import TextBlob


def getSubjectiveWordsCount(row):
    taggedText = nltk.pos_tag(nltk.word_tokenize(row.statement))
    # taggedText = nltk.pos_tag(nltk.word_tokenize(row.text))
    adwordsCount = 0
    for tag in taggedText:
        if (tag[1][0] == 'J') or (tag[1][0] == 'R'):
            adwordsCount += 1
    return adwordsCount


def getSubjectivity(row):
    subjectivity = TextBlob(row.statement).sentiment.subjectivity
    # subjectivity = TextBlob(row.text).sentiment.subjectivity
    if (subjectivity <= 0.3):
        return 'low'
    elif (subjectivity >= 0.7):
        return 'high'
    else:
        return 'medium'


def getPolarity(row):
    return TextBlob(row.statement).sentiment.polarity


def fakeTrue():
    # apply not working
    # TypeError: expected string or bytes-like object
    ftPath = 'cleanDatasets/clean_fake_true.csv'
    ft = pd.read_csv(ftPath)

    ft['subjectiveWordsCount'] = ft.apply(getSubjectiveWordsCount, axis=1)
    ft['subjectivity'] = ft.apply(getSubjectivity, axis=1)

    ft.to_csv(ftPath, encoding='utf-8-sig', index=False)


def liar():
    liarTrainPath = 'cleanDatasets/clean_liar_train.csv'
    liarTestPath = 'cleanDatasets/clean_liar_test.csv'
    liarValidPath = 'cleanDatasets/clean_liar_valid.csv'

    liarTrainData = pd.read_csv(liarTrainPath)
    liarTestData = pd.read_csv(liarTestPath)
    liarValidData = pd.read_csv(liarValidPath)

    liarTrainData['polarity'] = liarTrainData.apply(
        getPolarity, axis=1)
    liarTrainData['subjectiveWordsCount'] = liarTrainData.apply(
        getSubjectiveWordsCount, axis=1)
    liarTrainData['subjectivity'] = liarTrainData.apply(
        getSubjectivity, axis=1)

    liarTestData['polarity'] = liarTestData.apply(getPolarity, axis=1)
    liarTestData['subjectiveWordsCount'] = liarTestData.apply(
        getSubjectiveWordsCount, axis=1)
    liarTestData['subjectivity'] = liarTestData.apply(getSubjectivity, axis=1)

    liarValidData['polarity'] = liarValidData.apply(
        getPolarity, axis=1)
    liarValidData['subjectiveWordsCount'] = liarValidData.apply(
        getSubjectiveWordsCount, axis=1)
    liarValidData['subjectivity'] = liarValidData.apply(
        getSubjectivity, axis=1)

    liarTrainData.to_csv(liarTrainPath, encoding='utf-8-sig', index=False)
    liarTestData.to_csv(liarTestPath, encoding='utf-8-sig', index=False)
    liarValidData.to_csv(liarValidPath, encoding='utf-8-sig', index=False)


def main():
    # fakeTrue()
    liar()


main()

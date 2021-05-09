import nltk
from nltk import tokenize
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def getSentiment(row):  # Calculate sentiment analysis from Vader
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(row.statement)
    score.pop("compound")
    score.pop("neu")
    sentiment = max(score, key=score.get)
    return sentiment


def getSubjectivity(row):  # Calculate subjectivity rating from TextBlob and classify
    subjectivity = round(TextBlob(row.statement).sentiment.subjectivity, 2)
    # # subjectivity = round(TextBlob(row.text).sentiment.subjectivity,1)
    # return subjectivity

    if (subjectivity == 0.00):
        return "NONE"
    elif (subjectivity > 0.00) and (subjectivity <= 0.25):
        return "LOW"
    elif (subjectivity > 0.25) and (subjectivity <= 0.5):
        return "MEDIUM"
    elif (subjectivity > 0.5):
        return "HIGH"


def getPolarity(row):  # Calculate polarity rating from TextBlob
    polarity = TextBlob(row.statement).sentiment.polarity + 1
    # if polarity < 0:
    #     polarity = 0
    return polarity


def applySentimentToDF(df):  # Apply sentiment, polarity, and subjectivity to DataFrame
    df["polarity"] = df.apply(getPolarity, axis=1)
    df["subjectivity"] = df.apply(
        getSubjectivity, axis=1)
    df["sentiment"] = df.apply(getSentiment, axis=1)
    return df

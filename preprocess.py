import csv
import collections
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from subjectivity import applyToDF
# nltk.download('stopwords')
stop = stopwords.words('english')

# inspired from https://towardsdatascience.com/detecting-fake-news-with-and-without-code-dd330ed449d9


def punctuationRemoval(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


def simplifyLabel(df):
    df.loc[df.label == "true", "label"] = "TRUE"
    df.loc[df.label == "mostly-true", "label"] = "MOSTLY-TRUE"
    df.loc[df.label == "half-true", "label"] = "HALF-TRUE"
    df.loc[df.label == "barely-true", "label"] = "BARELY-TRUE"
    df.loc[df.label == "pants-fire", "label"] = "PANTS-FIRE"
    # df.loc[df.label == "pants-fire", "label"] = "FALSE"
    df.loc[df.label == "false", "label"] = "FALSE"
    return df


def handleNaN(df):
    for header in df.columns.values:
        df[header] = df[header].fillna(
            df[header][df[header].first_valid_index()])
    # df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    return df


def renameLiarColumn(df):
    df.rename(columns={'barely true counts': 'barelyTrueCounts',
                       'false counts': 'falseCounts', 'half true counts': 'halfTrueCounts', 'mostly true counts': 'mostlyTrueCounts', 'pants on fire counts': 'pantsOnFireCounts'})
    return df


def cleanDataText(df, textHeader):
    # make all lower case and apply for more than statement
    df[textHeader] = df[textHeader].str.lower()
    df[textHeader] = df[textHeader].apply(punctuationRemoval)
    df[textHeader] = df[textHeader].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop)]))
    df[textHeader] = df[textHeader].map(lambda x: re.sub(r'\W+', ' ', x))
    df = simplifyLabel(df)
    df = handleNaN(df)
    return df


def initLiarData():
    liar_train = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_train_labeled.csv').reset_index(drop=True), 'statement')
    liar_train = applyToDF(liar_train)
    liar_train.to_csv('./cleanDatasets/clean_liar_train.csv',
                      encoding='utf-8-sig', index=False)
    liar_test = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_test_labeled.csv').reset_index(drop=True), 'statement')
    liar_test = applyToDF(liar_test)
    liar_test.to_csv('./cleanDatasets/clean_liar_test.csv',
                     encoding='utf-8-sig', index=False)
    liar_valid = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_valid_labeled.csv').reset_index(drop=True), 'statement')
    liar_valid = applyToDF(liar_valid)
    liar_valid.to_csv('./cleanDatasets/clean_liar_valid.csv',
                      encoding='utf-8-sig', index=False)


def initPolitifact():
    data = cleanDataText(pd.read_csv(
        './datasets/PolitiFact/politifact.csv').reset_index(drop=True), 'statement')
    data = data.drop(['Unnamed: 0'], axis=1)
    data = data[data['label'] != 'full-flop']
    data = data[data['label'] != 'half-flip']
    data = data[data['label'] != 'no-flip']
    data = applyToDF(data)  # from subjectivity
    data.to_csv('./cleanDatasets/clean_politifact.csv', index=False)


def main():
    initLiarData()
    initPolitifact()


main()

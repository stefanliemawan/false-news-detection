import csv
import collections
import pandas as pd
from sklearn.utils import shuffle
import string
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
stop = stopwords.words('english')

# inspired from https://towardsdatascience.com/detecting-fake-news-with-and-without-code-dd330ed449d9


def punctuationRemoval(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


def simplifyLabel(df):
    # df.loc[df.label == "mostly-true", "label"] = "TRUE"
    # df.loc[df.label == "half-true", "label"] = "HALF-TRUE"
    # df.loc[df.label == "barely-true", "label"] = "HALF-TRUE"
    # df.loc[df.label == "pants-fire", "label"] = "FALSE"
    df.loc[df.label == "mostly-true", "label"] = "MOSTLY-TRUE"
    df.loc[df.label == "half-true", "label"] = "HALF-TRUE"
    df.loc[df.label == "barely-true", "label"] = "BARELY-TRUE"
    df.loc[df.label == "pants-fire", "label"] = "FALSE"
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
    df = simplifyLabel(df)
    df = handleNaN(df)
    return df


def initLiarData():
    liar_train = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_train_labeled.csv').reset_index(drop=True), 'statement')
    liar_train.to_csv('./cleanDatasets/clean_liar_train.csv',
                      encoding='utf-8-sig', index=False)
    liar_test = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_test_labeled.csv').reset_index(drop=True), 'statement')
    liar_test.to_csv('./cleanDatasets/clean_liar_test.csv',
                     encoding='utf-8-sig', index=False)
    liar_valid = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_valid_labeled.csv').reset_index(drop=True), 'statement')
    liar_valid.to_csv('./cleanDatasets/clean_liar_valid.csv',
                      encoding='utf-8-sig', index=False)


def main():
    initLiarData()


main()

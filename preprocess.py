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

# punctuationRemoval and cleanDataText might not be needed cause scikit learn has stop words(?)


def cleanDataText(data, textHeader):
    data[textHeader] = data[textHeader].apply(punctuationRemoval)
    data[textHeader] = data[textHeader].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop)]))
    return data


def initFakeTrueData():
    fake = pd.read_csv('./datasets/FakeTrue/Fake.csv')
    true = pd.read_csv('./datasets/FakeTrue/True.csv')
    fake['label'] = 'fake'
    true['label'] = 'true'
    data = pd.concat([fake, true]).reset_index(drop=True)
    data = shuffle(data)
    data = data.reset_index(drop=True)
    cleanData = cleanDataText(
        data, 'title')
    cleanData = cleanDataText(
        data, 'text')
    cleanData.to_csv('./cleanDatasets/clean_fake_true.csv',
                     encoding='utf-8-sig')


def renameLiarColumn(dataframe):
    dataframe.rename(columns={'barely true counts': 'barelyTrueCounts',
                              'false counts': 'falseCounts', 'half true counts': 'halfTrueCounts', 'mostly true counts': 'mostlyTrueCounts', 'pants on fire counts': 'pantsOnFireCounts'})
    return dataframe


def initLiarData():
    liarTrain = cleanDataText(shuffle(pd.read_csv('./datasets/LIAR/liar_train_labeled.csv')
                                      ).reset_index(drop=True), 'statement')
    liarTrain.to_csv('./cleanDatasets/clean_liar_train.csv',
                     encoding='utf-8-sig', index=False)
    liarTest = cleanDataText(shuffle(pd.read_csv('./datasets/LIAR/liar_test_labeled.csv')
                                     ).reset_index(drop=True), 'statement')
    liarTest.to_csv('./cleanDatasets/clean_liar_test.csv',
                    encoding='utf-8-sig', index=False)
    liarValid = cleanDataText(shuffle(pd.read_csv('./datasets/LIAR/liar_valid_labeled.csv')
                                      ).reset_index(drop=True), 'statement')
    liarValid.to_csv('./cleanDatasets/clean_liar_valid.csv',
                     encoding='utf-8-sig', index=False)


def main():
    # initFakeTrueData()
    initLiarData()


main()

import csv
import collections
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from subjectivity import applyToDF
from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
# nltk.download('stopwords')
# nltk.download('tagsets')
# nltk.download('wordnet')
stop = stopwords.words('english')

# inspired from https://towardsdatascience.com/detecting-fake-news-with-and-without-code-dd330ed449d9


pd.set_option('display.max_colwidth', None)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# stemmer = SnowballStemmer("english")


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
    df.loc[df.label == "false", "label"] = "FALSE"
    df.loc[df.label == "true", "label"] = "TRUE"

    # df.loc[df.label == "true", "label"] = "TRUE"
    # df.loc[df.label == "mostly-true", "label"] = "TRUE"
    # df.loc[df.label == "half-true", "label"] = "HALF-TRUE"
    # df.loc[df.label == "barely-true", "label"] = "HALF-TRUE"
    # df.loc[df.label == "pants-fire", "label"] = "FALSE"
    # df.loc[df.label == "false", "label"] = "FALSE"
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


def lemmatize(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


def stem(text):
    return ' '.join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])


def addTags(data):
    # tags = set(['', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN',
    #             'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'])
    # should use df.apply?
    # nltk.help.upenn_tagset()
    for index, row in data.iterrows():
        sentence = nltk.pos_tag(nltk.word_tokenize(str(row['statement'])))
        count = Counter([j for i, j in sentence])
        for tag in list(count):
            data.at[index, tag] = count[tag]
    data.fillna(0, inplace=True)
    return data


def cleanDataText(df):
    # remove number?
    df = addTags(df)
    df['statement'] = df['statement'].str.lower()
    df['statement'] = df['statement'].str.replace('.', '', regex=False)
    print(df['statement'].iloc[1500:1510])
    df['statement'] = df['statement'].map(lambda x: re.sub(
        r'\W+', ' ', x))  # remove non-words character
    print(df['statement'].iloc[1500:1510])
    df['statement'] = df['statement'].map(
        lambda x: re.sub(r'\d+', ' ', x))  # remove numbers
    print(df['statement'].iloc[1500:1510])
    df['statement'] = df['statement'].map(
        lambda x: re.sub(r'\s+', ' ', x).strip())  # remove double space
    print(df['statement'].iloc[1500:1510])
    df['statement'] = df['statement'].apply(punctuationRemoval)
    print(df['statement'].iloc[1500:1510])
    df['statement'] = df['statement'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop)]))  # remove stop words

    print(df['statement'].iloc[1500:1510])
    df['statement'] = df['statement'].apply(lemmatize)
    print(df['statement'].iloc[1500:1510])
    # df['statement'] = df['statement'].apply(stem)
    # print(df['statement'].iloc[1500:1510])
    # print(a)
    df = simplifyLabel(df)
    # df = handleNaN(df)
    return df


def initLiarData():
    liar_train = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_train_labeled.csv').reset_index(drop=True))
    liar_train = applyToDF(liar_train)
    liar_train.to_csv('./cleanDatasets/clean_liar_train.csv',
                      encoding='utf-8-sig', index=False)
    liar_test = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_test_labeled.csv').reset_index(drop=True))
    liar_test = applyToDF(liar_test)
    liar_test.to_csv('./cleanDatasets/clean_liar_test.csv',
                     encoding='utf-8-sig', index=False)
    liar_valid = cleanDataText(pd.read_csv(
        './datasets/LIAR/liar_valid_labeled.csv').reset_index(drop=True))
    liar_valid = applyToDF(liar_valid)
    liar_valid.to_csv('./cleanDatasets/clean_liar_valid.csv',
                      encoding='utf-8-sig', index=False)


def initPolitifact():
    data = cleanDataText(pd.read_csv(
        './datasets/PolitiFact/politifact.csv').reset_index(drop=True))
    data = data.drop(['Unnamed: 0'], axis=1)
    data = data[data['label'] != 'full-flop']

    data = data[data['label'] != 'half-flip']
    data = data[data['label'] != 'no-flip']
    data = applyToDF(data)  # from subjectivity
    data.to_csv('./cleanDatasets/clean_politifact.csv', index=False)


def initMergedPolitifact():
    data = pd.read_csv(
        './datasets/merged_politifact.csv').reset_index(drop=True)
    data = applyToDF(data)  # from subjectivity
    data = cleanDataText(data)
    data['speaker'] = data['speaker'].str.lower()
    data['speaker'] = data['speaker'].str.replace(' ', '-')
    data['speaker'] = data['speaker'].str.replace('"', '')

    data = data[data['label'] != 'full-flop']
    data = data[data['label'] != 'half-flip']
    data = data[data['label'] != 'no-flip']
    data = data.drop_duplicates(subset="statement")
    data.dropna(inplace=True)
    print(data)
    print(data.shape)

    # print(data.head())
    # is_NaN = data.isnull()
    # row_has_NaN = is_NaN.any(axis=1)
    # rows_with_NaN = data[row_has_NaN]
    # print(rows_with_NaN)

    print(data['label'].value_counts())
    print(data['subjectivity'].value_counts())

    data.to_csv('./cleanDatasets/clean_merged_politifact.csv', index=False)
    # data.to_csv('./cleanDatasets/clean_merged_politifact_stemmed.csv', index=False)


def initFakeNewsNet():
    politi_fake = cleanDataText(pd.read_csv(
        './datasets/FakeNewsNet/politifact_fake.csv').reset_index(drop=True), 'title')
    politi_real = cleanDataText(pd.read_csv(
        './datasets/FakeNewsNet/politifact_real.csv').reset_index(drop=True), 'title')
    gossip_fake = cleanDataText(pd.read_csv(
        './datasets/FakeNewsNet/gossipcop_fake.csv').reset_index(drop=True), 'title')
    gossip_real = cleanDataText(pd.read_csv(
        './datasets/FakeNewsNet/gossipcop_real.csv').reset_index(drop=True), 'title')

    politi_real['label'] = "TRUE"
    politi_fake['label'] = "FALSE"
    gossip_real['label'] = "TRUE"
    gossip_fake['label'] = "FALSE"

    politi = pd.concat([politi_real, politi_fake], ignore_index=True)
    gossip = pd.concat([gossip_real, gossip_fake], ignore_index=True)

    print(politi)
    print(gossip)

    politi.to_csv('./cleanDatasets/FNN_politifact.csv', index=False)
    gossip.to_csv('./cleanDatasets/FNN_gossip.csv', index=False)


def mergePolitifact():
    liar_train = pd.read_csv(
        './cleanDatasets/clean_liar_train.csv').reset_index(drop=True)
    liar_test = pd.read_csv(
        './cleanDatasets/clean_liar_test.csv').reset_index(drop=True)
    liar_val = pd.read_csv(
        './cleanDatasets/clean_liar_valid.csv').reset_index(drop=True)
    politi = pd.read_csv(
        './cleanDatasets/clean_politifact.csv').reset_index(drop=True)
    data = pd.concat([liar_train, liar_test, liar_val,
                      politi])
    print(data.shape)
    data = data.drop_duplicates(subset="statement")
    data = data.reset_index(drop=True)
    data = data.drop(['id', 'subject', "speaker's job title",
                      'state', 'party', 'context', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', 'date', 'checker'], axis=1)
    data['speaker'] = data['speaker'].str.lower()
    data['speaker'] = data['speaker'].str.replace('-', ' ')
    print(data)
    print(data.shape)
    data.to_csv('./cleanDatasets/merged_politifact.csv', index=False)


def main():
    # inplace true?
    # initLiarData()
    # initPolitifact()
    # mergePolitifact()
    initMergedPolitifact()
    # initFakeNewsNet()


main()

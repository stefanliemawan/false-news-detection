
from subjectivity import applySentimentToDF
import csv
import collections
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from sklearn.utils import shuffle
# nltk.download("stopwords")
# nltk.download("tagsets")
# nltk.download("wordnet")
stop = stopwords.words("english")


pd.set_option("display.max_colwidth", None)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def punctuationRemoval(text):  # Remove punctuation from text
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = "".join(all_list)
    return clean_str


def simplifyLabel(data):  # Handle and unify all label
    data.loc[data.label == "true", "label"] = "TRUE"
    data.loc[data.label == "mostly-true", "label"] = "MOSTLY-TRUE"
    data.loc[data.label == "half-true", "label"] = "HALF-TRUE"
    data.loc[data.label == "barely-true", "label"] = "MOSTLY-FALSE"
    data.loc[data.label == "false", "label"] = "FALSE"
    data.loc[data.label == "pants-fire", "label"] = "PANTS-ON-FIRE"

    data.loc[data.label == "full-flop", "label"] = "FALSE"
    data.loc[data.label == "half-flip", "label"] = "HALF-TRUE"
    data.loc[data.label == "no-flip", "label"] = "TRUE"
    return data


def lemmatize(text):  # Lemmatize text
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


def stem(text):  # Stem text
    return " ".join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])


def addTags(data):  # Add POS TAGS to DataFrame
    tags = sorted(set(["CC", "CD", "DT", "IN",
                       "JJ", "JJR", "JJS", "MD", "NN", "NNS", "NNP", "NNPS", "PRP", "PRP$", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))
    for t in tags:
        data[t] = 0

    for index, row in data.iterrows():
        sentence = nltk.pos_tag(nltk.word_tokenize(str(row["statement"])))
        count = Counter([j for i, j in sentence])
        for tag in list(count):
            if tag in tags:
                data.at[index, tag] = int(count[tag])
    data.fillna(0, inplace=True)
    return data


# Add Apostrophe to contraction words so that they are recognized as stop words (especially in LIAR)
def cleanContractions(data):
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bisnt\b", "isn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\barent\b", "aren't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bwasnt\b", "wasn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bwerent\b", "weren't", x))

    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bcant\b", "can't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bcouldnt\b", "couldn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bshouldnt\b", "shouldn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bwouldnt\b", "wouldn't", x))

    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bhasnt\b", "hasn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bhadnt\b", "hadn't", x))

    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bdoesnt\b", "doesn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bdont\b", "don't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bdidnt\b", "didn't", x))

    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bmustnt\b", "mustn't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bwont\b", "won't", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bshant\b", "shan't", x))
    # 're
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bim\b", "i'm", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\byoure\b", "you're", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\btheyre\b", "they're", x))
    # ll
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\byoull\b", "you'll", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\btheyll\b", "they'll", x))
    # ve
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\byouve\b", "you've", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bive\b", "i've", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bweve\b", "we've", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\btheyve\b", "they've", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bcouldve\b", "could've", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bshouldve\b", "should've", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bwouldve\b", "would've", x))
    # s
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bshes\b", "she's", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bhes\b", "he's", x))
    # d
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\byoud\b", "you'd", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bhed\b", "he'd", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bthad\b", "that'd", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\btheyd\b", "they'd", x))
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\bwed\b", "we'd", x))
    return data


def cleanDataText(data):  # Preprocess text
    data = simplifyLabel(data)
    data = addTags(data)

    data["raw"] = data["statement"]
    data["statement"] = data["statement"].str.lower()
    data = cleanContractions(data)
    data["statement"] = data["statement"].str.replace(".", "", regex=False)
    data["statement"] = data["statement"].str.replace(
        "â€™", " ", regex=False)  # this character appears in some rows
    data["statement"] = data["statement"].str.replace("'", " ", regex=False)
    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].map(lambda x: re.sub(
        r"\W+", " ", x))  # remove non-words character
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\d+", " ", x))  # remove numbers
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\s+", " ", x).strip())  # remove double space
    data["statement"] = data["statement"].apply(punctuationRemoval)
    data["statement"] = data["statement"].apply(lambda x: " ".join(
        [word for word in x.split() if word not in (stop)]))  # remove stop words

    data["statement"] = data["statement"].apply(lemmatize)
    # data["statement"] = data["statement"].apply(stem)
    # print(data["statement"].iloc[1500:1510])

    data["speaker"] = data["speaker"].str.replace("-", " ", regex=False)
    data["speaker"] = data["speaker"].str.replace("'", "", regex=False)
    data["speaker"] = data["speaker"].str.title()

    data["subject"] = data["subject"].str.title()
    data["party"] = data["party"].str.title()

    data["speaker's job title"] = data["speaker's job title"].str.replace(
        ".", "", regex=False)
    data["speaker's job title"] = data["speaker's job title"].str.title()
    data["context"] = data["context"].str.replace(
        ".", "", regex=False)
    return data


def substractCount(data): # Substract current label from the history count + calculate credit score
    for index, row in data.iterrows():
        label = row.label.lower()
        label = label.replace("-", " ")
        col = label + " counts"
        if col in data.columns and row[col] > 0:
            data.loc[index, col] -= 1

        mt = row["mostly true counts"]
        ht = row["half true counts"]
        mf = row["mostly false counts"]
        f = row["false counts"]
        pf = row["pants on fire counts"]
        div = (mt + ht + mf + f + pf)
        if div == 0:
            score = 0.5
        else:
            score = (mt*0.2 + ht*0.5 + mf*0.75 + f *
                     0.9 + pf*1) / div
        data.loc[index, "credit score"] = score
    return data


def initLIAR():  # initialize LIAR dataset
    liar_train = pd.read_csv(
        "./datasets/LIAR/liar_train_labeled.csv").reset_index(drop=True)
    liar_train = applySentimentToDF(liar_train)  # from subjectivity
    liar_train = cleanDataText(liar_train)
    liar_train.rename(
        columns={"barely true counts": "mostly false counts"}, inplace=True)
    liar_train = substractCount(liar_train)
    print(liar_train.loc[0])
    liar_train.to_csv(
        "./cleanDatasets/clean_liar_train.csv", index=False)
    liar_train = fillMissingMetadata(liar_train)
    liar_train.to_csv(
        "./cleanDatasets/clean_liar_train+.csv", index=False)

    liar_test = pd.read_csv(
        "./datasets/LIAR/liar_test_labeled.csv").reset_index(drop=True)
    liar_test = applySentimentToDF(liar_test)  # from subjectivity
    liar_test = cleanDataText(liar_test)
    liar_test.rename(
        columns={"barely true counts": "mostly false counts"}, inplace=True)
    liar_test = substractCount(liar_test)
    liar_test.to_csv(
        "./cleanDatasets/clean_liar_test.csv", index=False)
    liar_test = fillMissingMetadata(liar_test)
    liar_test.to_csv(
        "./cleanDatasets/clean_liar_test+.csv", index=False)

    liar_val = pd.read_csv(
        "./datasets/LIAR/liar_valid_labeled.csv").reset_index(drop=True)
    liar_val = applySentimentToDF(liar_val)  # from subjectivity
    liar_val = cleanDataText(liar_val)
    liar_val.rename(
        columns={"barely true counts": "mostly false counts"}, inplace=True)
    liar_val = substractCount(liar_val)
    liar_val.to_csv(
        "./cleanDatasets/clean_liar_valid.csv", index=False)
    liar_val = fillMissingMetadata(liar_val)
    liar_val.to_csv(
        "./cleanDatasets/clean_liar_valid+.csv", index=False)


def mergePoliti():  # Merge scraped dataset with LIAR
    liar_train = pd.read_csv(
        "./datasets/LIAR/liar_train_labeled.csv").reset_index(drop=True)
    liar_test = pd.read_csv(
        "./datasets/LIAR/liar_test_labeled.csv").reset_index(drop=True)
    liar_val = pd.read_csv(
        "./datasets/LIAR/liar_valid_labeled.csv").reset_index(drop=True)
    politi = pd.read_csv(
        "./datasets/PolitiFact/politifact.csv").reset_index(drop=True)
    liar = pd.concat([liar_train, liar_test, liar_val])
    liar.rename(
        columns={"barely true counts": "mostly false counts"}, inplace=True)
    data = pd.DataFrame(
        columns=["label", "statement", "subject", "speaker", "speaker's job title", "state", "party", "true counts", "mostly true counts", "half true counts", "mostly false counts", "false counts", "pants on fire counts", "context"])
    data = pd.concat([data, liar, politi], ignore_index=True)
    data.drop_duplicates(subset="statement", keep='last', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.drop(["id"], axis=1, inplace=True)
    data.to_csv("./datasets/merged_politifact.csv", index=False)


def initMergedPoliti():  # Initialize POLITI dataset
    data = pd.read_csv(
        "./datasets/merged_politifact.csv").reset_index(drop=True)
    data = applySentimentToDF(data)  # from subjectivity
    data = cleanDataText(data)
    data.drop_duplicates(subset="statement", keep='last', inplace=True)

    print(data["label"].value_counts())
    print(data["subjectivity"].value_counts())

    data.to_csv("./cleanDatasets/clean_merged_politifact.csv", index=False)
    data = fillMissingMetadata(data)
    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+.csv", index=False)


# use speaker-profile (profiles.csv) to fill values in parameter dataset
def fillMissingMetadata(data):
    profiles = pd.read_csv("./cleanDatasets/profiles.csv")
    # data.sort_values(by=["speaker", "speaker's job title",
    #                      "state", "party", "true counts", "mostly true counts",	"half true counts",	"mostly false counts", "false counts", "pants on fire counts"], inplace=True)
    # data.sort_values(by=["speaker"], inplace=True)
    # profiles.sort_values(by=["speaker"], inplace=True)
    data["speaker"] = data["speaker"].str.title()
    profiles["speaker"] = profiles["speaker"].str.title()

    if "true counts" not in data.columns:
        data["true counts"] = pd.Series(None, index=data.index)
    data = data.merge(profiles, how="left", on="speaker", suffixes=("_x", ""))

    data["speaker's job title"].fillna(
        data["speaker's job title_x"], inplace=True)
    data.drop(["speaker's job title_x"], axis=1, inplace=True)

    data["state"].fillna(data["state_x"], inplace=True)
    data.drop(["state_x"], axis=1, inplace=True)
    data["party"].fillna(data["party_x"], inplace=True)
    data.drop(["party_x"], axis=1, inplace=True)
    data["true counts"].fillna(data["true counts_x"], inplace=True)
    data.drop(["true counts_x"], axis=1, inplace=True)
    data["mostly true counts"].fillna(
        data["mostly true counts_x"], inplace=True)
    data.drop(["mostly true counts_x"], axis=1, inplace=True)
    data["half true counts"].fillna(data["half true counts_x"], inplace=True)
    data.drop(["half true counts_x"], axis=1, inplace=True)
    data["mostly false counts"].fillna(
        data["mostly false counts_x"], inplace=True)
    data.drop(["mostly false counts_x"], axis=1, inplace=True)
    data["false counts"].fillna(data["false counts_x"], inplace=True)
    data.drop(["false counts_x"], axis=1, inplace=True)
    data["pants on fire counts"].fillna(
        data["pants on fire counts_x"], inplace=True)
    data.drop(["pants on fire counts_x"], axis=1, inplace=True)
    if "count_x" in data.columns:
        data.drop(["count_x"], axis=1, inplace=True)

    data = data.groupby(["speaker"], as_index=False).apply(
        lambda group: group.ffill())
    data = substractCount(data)

    data = shuffle(data)
    return data


def fillProfilesWithLIARState():  # Use LIAR State information to fill speaker-profile
    liar_train = pd.read_csv(
        "./datasets/LIAR/liar_train_labeled.csv").reset_index(drop=True)
    liar_test = pd.read_csv(
        "./datasets/LIAR/liar_test_labeled.csv").reset_index(drop=True)
    liar_val = pd.read_csv(
        "./datasets/LIAR/liar_valid_labeled.csv").reset_index(drop=True)
    liar = pd.concat([liar_train, liar_test, liar_val])
    profiles = pd.read_csv(
        "./cleanDatasets/profiles.csv").reset_index(drop=True)
    liar = liar[["speaker", "speaker's job title", "state", "party"]]
    liar["speaker"] = liar["speaker"].str.title()
    liar["speaker"] = liar["speaker"].str.replace("-", " ")
    liar.sort_values(by=["speaker"], inplace=True)
    liar.drop_duplicates(subset=["speaker"], inplace=True)

    profiles = profiles.merge(
        liar[["speaker", "state"]], how="left", on="speaker", suffixes=("_x", ""))
    profiles["state"].fillna(profiles["state_x"], inplace=True)
    profiles.drop(["state_x"], axis=1, inplace=True)

    profiles.to_csv(
        "./cleanDatasets/profiles.csv", index=False)


def sortByCount():  # Sort dataset by count, debugging and data viewing purpose only
    data = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact+_editeds.csv").reset_index(drop=True)
    profiles = pd.read_csv(
        "./cleanDatasets/profiles.csv").reset_index(drop=True)

    data["speaker"] = data["speaker"].str.title()
    profiles["speaker"] = profiles["speaker"].str.title()

    data["count"] = data.groupby(["speaker"])["speaker"].transform("count")
    data.sort_values(by=["count", "speaker"], ascending=False, inplace=True)
    print(data[["speaker", "count"]].head())

    profiles = profiles.merge(
        data[["speaker", "count"]], how="left", on="speaker", suffixes=("_x", ""))
    profiles.sort_values(by=["count", "speaker"],
                         ascending=False, inplace=True)
    profiles.drop_duplicates(subset="speaker", inplace=True)

    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+_editeds.csv", index=False)

    profiles.to_csv(
        "./cleanDatasets/profiles.csv", index=False)


def updateDataset():  # Update dataset after manually editing speaker-profile
    profiles = pd.read_csv(
        "./cleanDatasets/profiles.csv").reset_index(drop=True)
    print(profiles.isna().sum())
    print("")

    politi = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact.csv").reset_index(drop=True)
    politi = fillMissingMetadata(politi)
    politi.to_csv(
        "./cleanDatasets/clean_merged_politifact+.csv", index=False)

    liar_train = pd.read_csv(
        "./cleanDatasets/clean_liar_train.csv").reset_index(drop=True)
    liar_train = fillMissingMetadata(liar_train)
    liar_train.to_csv(
        "./cleanDatasets/clean_liar_train+.csv", index=False)

    liar_test = pd.read_csv(
        "./cleanDatasets/clean_liar_test.csv").reset_index(drop=True)
    liar_test = fillMissingMetadata(liar_test)
    liar_test.to_csv(
        "./cleanDatasets/clean_liar_test+.csv", index=False)

    liar_val = pd.read_csv(
        "./cleanDatasets/clean_liar_valid.csv").reset_index(drop=True)
    liar_val = fillMissingMetadata(liar_val)
    liar_val.to_csv(
        "./cleanDatasets/clean_liar_valid+.csv", index=False)


def main():
    initLIAR()
    mergePoliti()
    initMergedPoliti()

    # fillProfilesWithLIARState()
    # updateDataset()()


main()

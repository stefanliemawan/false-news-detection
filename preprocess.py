
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
# stemmer = SnowballStemmer("english")


def punctuationRemoval(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = "".join(all_list)
    return clean_str


def simplifyLabel(data):
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


def handleNaN(data):
    for header in data.columns.values:
        data[header] = data[header].fillna(
            data[header][data[header].first_valid_index()])
    return data


def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


def stem(text):
    return " ".join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])


def addTags(data):
    tags = sorted(set(["CC", "CD", "DT", "IN",
                       "JJ", "JJR", "JJS", "MD", "NN", "NNS", "NNP", "NNPS", "PRP", "PRP$", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))
    for t in tags:
        data[t] = 0
    # should use data.apply?
    # nltk.help.upenn_tagset()

    for index, row in data.iterrows():
        sentence = nltk.pos_tag(nltk.word_tokenize(str(row["statement"])))
        count = Counter([j for i, j in sentence])
        for tag in list(count):
            if tag in tags:
                data.at[index, tag] = int(count[tag])
    data.fillna(0, inplace=True)
    print(data.columns.values)
    return data


def cleanContractions(data):
    # for LIAR
    # nt
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


def cleanDataText(data):
    data = simplifyLabel(data)
    data = addTags(data)

    data["raw"] = data["statement"]
    data["statement"] = data["statement"].str.lower()
    data = cleanContractions(data)
    data["statement"] = data["statement"].str.replace(".", "", regex=False)
    data["statement"] = data["statement"].str.replace("â€™", " ", regex=False)
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

    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].apply(lemmatize)
    print(data["statement"].iloc[1500:1510])
    # data["statement"] = data["statement"].apply(stem)
    # print(data["statement"].iloc[1500:1510])
    # print(a)
    # data = handleNaN(data)

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


def substractCount(data):
    for index, row in data.iterrows():
        label = row.label.lower()
        label = label.replace("-", " ")
        col = label + " counts"
        if col in data.columns and row[col] > 0:
            data.loc[index, col] -= 1
    return data


def initLIAR():
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


def mergePolitifact():
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
    print(data)
    print(data.columns.values)
    print(data.shape)
    data.to_csv("./datasets/merged_politifact.csv", index=False)


def initMergedPolitifact():
    data = pd.read_csv(
        "./datasets/merged_politifact.csv").reset_index(drop=True)
    data = applySentimentToDF(data)  # from subjectivity
    data = cleanDataText(data)
    data.drop_duplicates(subset="statement", keep='last', inplace=True)
    # 312 row with NaN
    # data.dropna(inplace=True)
    # print(data.isna().sum().sum())
    print(data)
    print(data.shape)

    print(data["label"].value_counts())
    print(data["subjectivity"].value_counts())

    data.to_csv("./cleanDatasets/clean_merged_politifact.csv", index=False)
    data = fillMissingMetadata(data)
    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+.csv", index=False)


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
    # print(data.columns.values)

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
    # fill with other columns

    data = data.groupby(["speaker"], as_index=False).apply(
        lambda group: group.ffill())
    data = substractCount(data)
    # print(data[["speaker", "speaker's job title"]].iloc[:50])

    print(data.columns.values)
    data = shuffle(data)
    return data


def fillCountMetadata():
    data = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact+_edited.csv").reset_index(drop=True)
    profiles = pd.read_csv(
        "./cleanDatasets/profiles.csv").reset_index(drop=True)

    count_cols = ["speaker", "true counts", "mostly true counts",
                  "half true counts", "mostly false counts", "false counts", "pants on fire counts"]

    data = data.merge(
        profiles[count_cols], how="left", on="speaker", suffixes=("_x", ""))
    print(data.columns.values)
    data["true counts"].fillna(data["true counts_x"], inplace=True)
    data["mostly true counts"].fillna(
        data["mostly true counts_x"], inplace=True)
    data["half true counts"].fillna(data["half true counts_x"], inplace=True)
    data["mostly false counts"].fillna(
        data["mostly false counts_x"], inplace=True)
    data["false counts"].fillna(data["false counts_x"], inplace=True)
    data["pants on fire counts"].fillna(
        data["pants on fire counts_x"], inplace=True)

    data.drop(["true counts_x", "mostly true counts_x",
               "half true counts_x", "mostly false counts_x", "false counts_x", "pants on fire counts_x"], axis=1, inplace=True)
    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+_edited.csv", index=False)


def fillProfilesWithLIARState():
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
    print(profiles[["speaker", "state_x", "state"]].loc[30:60])
    profiles.drop(["state_x"], axis=1, inplace=True)

    print(profiles.isna().sum())
    print(profiles.shape)

    profiles.to_csv(
        "./cleanDatasets/profiles.csv", index=False)


def fillMergedPolitiFactWithProfiles():
    data = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact+_edited.csv").reset_index(drop=True)
    profiles = pd.read_csv(
        "./datasets/PolitiFact/profiles.csv").reset_index(drop=True)
    data = data.merge(profiles[["speaker", "state"]],
                      how="left", on="speaker", suffixes=("_x", ""))
    print(profiles.columns.values)
    data["state"].fillna(data["state_x"], inplace=True)
    print(data[["speaker", "state_x", "state"]].loc[60:100])
    print(data[["speaker", "state_x", "state"]].loc[120:300])
    data.drop(["state_x"], axis=1, inplace=True)

    print(data.isna().sum())
    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+_edited.csv", index=False)


def initFakeNewsNet():
    politi_fake = cleanDataText(pd.read_csv(
        "./datasets/FakeNewsNet/politifact_fake.csv").reset_index(drop=True), "title")
    politi_real = cleanDataText(pd.read_csv(
        "./datasets/FakeNewsNet/politifact_real.csv").reset_index(drop=True), "title")
    gossip_fake = cleanDataText(pd.read_csv(
        "./datasets/FakeNewsNet/gossipcop_fake.csv").reset_index(drop=True), "title")
    gossip_real = cleanDataText(pd.read_csv(
        "./datasets/FakeNewsNet/gossipcop_real.csv").reset_index(drop=True), "title")

    politi_real["label"] = "TRUE"
    politi_fake["label"] = "FALSE"
    gossip_real["label"] = "TRUE"
    gossip_fake["label"] = "FALSE"

    politi = pd.concat([politi_real, politi_fake], ignore_index=True)
    gossip = pd.concat([gossip_real, gossip_fake], ignore_index=True)

    print(politi)
    print(gossip)

    politi.to_csv("./cleanDatasets/FNN_politifact.csv", index=False)
    gossip.to_csv("./cleanDatasets/FNN_gossip.csv", index=False)


def sortByCount():
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


def main():
    initLIAR()
    # mergePolitifact()
    # initMergedPolitifact()

    # speaker2credit()
    # fillProfilesWithLIARState()
    # fillMergedPolitiFactWithProfiles()
    # initFakeNewsNet()

    # profiles = pd.read_csv(
    #     "./cleanDatasets/profiles.csv").reset_index(drop=True)
    # print(profiles.isna().sum())
    # print("")

    # politi = pd.read_csv(
    #     "./cleanDatasets/clean_merged_politifact.csv").reset_index(drop=True)
    # politi = fillMissingMetadata(politi)
    # politi.to_csv(
    #     "./cleanDatasets/clean_merged_politifact+.csv", index=False)

    # sorted_politi = pd.read_csv(
    #     "./cleanDatasets/clean_merged_politifact+_editeds.csv").reset_index(drop=True)
    # sorted_politi = fillMissingMetadata(sorted_politi)
    # sorted_politi.to_csv(
    #     "./cleanDatasets/clean_merged_politifact+_editeds_shuffled.csv", index=False)
    # sorted_politi.sort_values(
    #     by=["count", "speaker"], ascending=False, inplace=True)
    # sorted_politi.to_csv(
    #     "./cleanDatasets/clean_merged_politifact+_editeds.csv", index=False)

    # liar_train = pd.read_csv(
    #     "./cleanDatasets/clean_liar_train.csv").reset_index(drop=True)
    # liar_train = fillMissingMetadata(liar_train)
    # liar_train.to_csv(
    #     "./cleanDatasets/clean_liar_train+.csv", index=False)

    # liar_test = pd.read_csv(
    #     "./cleanDatasets/clean_liar_test.csv").reset_index(drop=True)
    # liar_test = fillMissingMetadata(liar_test)
    # liar_test.to_csv(
    #     "./cleanDatasets/clean_liar_test+.csv", index=False)

    # liar_val = pd.read_csv(
    #     "./cleanDatasets/clean_liar_valid.csv").reset_index(drop=True)
    # liar_val = fillMissingMetadata(liar_val)
    # liar_val.to_csv(
    #     "./cleanDatasets/clean_liar_valid+.csv", index=False)

    # clean subject as well?

    # data = pd.read_csv(
    #     "./cleanDatasets/clean_merged_politifact+_editeds.csv").reset_index(drop=True)
    # data = shuffle(data)
    # data.to_csv(
    #     "./cleanDatasets/clean_merged_politifact+_editeds_shuffled.csv", index=False)


main()

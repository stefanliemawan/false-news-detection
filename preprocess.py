import csv
import collections
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from subjectivity import applyToDF
from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
# nltk.download("stopwords")
# nltk.download("tagsets")
# nltk.download("wordnet")
stop = stopwords.words("english")

# inspired from https://towardsdatascience.com/detecting-fake-news-with-and-without-code-dd330ed449d9


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
    data.loc[data.label == "pants-fire", "label"] = "PANTS-FIRE"

    # data.loc[data.label == "true", "label"] = "TRUE"
    # data.loc[data.label == "mostly-true", "label"] = "TRUE"
    # data.loc[data.label == "half-true", "label"] = "HALF-TRUE"
    # data.loc[data.label == "barely-true", "label"] = "HALF-TRUE"
    # data.loc[data.label == "pants-fire", "label"] = "FALSE"
    # data.loc[data.label == "false", "label"] = "FALSE"
    return data


def handleNaN(data):
    for header in data.columns.values:
        data[header] = data[header].fillna(
            data[header][data[header].first_valid_index()])
    # data = data.interpolate(method="linear", limit_direction="forward", axis=0)
    return data


def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


def stem(text):
    return " ".join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])


def addTags(data):
    # tags = set(["", "CC", "CD", "DT", "EX", "FW", "IN",
    #             "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"])
    # should use data.apply?
    # nltk.help.upenn_tagset()
    for index, row in data.iterrows():
        sentence = nltk.pos_tag(nltk.word_tokenize(str(row["statement"])))
        count = Counter([j for i, j in sentence])
        for tag in list(count):
            data.at[index, tag] = count[tag]
    data.fillna(0, inplace=True)
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
        lambda x: re.sub(r"\bcouldnt\b", "could't", x))
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
    # data = addTags(data)
    data["statement"] = data["statement"].str.lower()
    data = cleanContractions(data)
    data["statement"] = data["statement"].str.replace(".", "", regex=False)
    data["statement"] = data["statement"].str.replace("â€™", " ", regex=False)
    data["statement"] = data["statement"].str.replace("'", " ", regex=False)
    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].map(lambda x: re.sub(
        r"\W+", " ", x))  # remove non-words character
    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\d+", " ", x))  # remove numbers
    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].map(
        lambda x: re.sub(r"\s+", " ", x).strip())  # remove double space
    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].apply(punctuationRemoval)
    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].apply(lambda x: " ".join(
        [word for word in x.split() if word not in (stop)]))  # remove stop words

    print(data["statement"].iloc[1500:1510])
    data["statement"] = data["statement"].apply(lemmatize)
    print(data["statement"].iloc[1500:1510])
    # data["statement"] = data["statement"].apply(stem)
    # print(data["statement"].iloc[1500:1510])
    # print(a)
    data = simplifyLabel(data)
    # data = handleNaN(data)
    return data


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
    # data.drop(["id", "speaker's job title", "true counts",
    #            "state", "party", "date", "Unnamed: 0"], axis=1, inplace=True)
    data.drop(["id", "date", "Unnamed: 0"], axis=1, inplace=True)
    print(data)
    print(data.columns.values)
    print(data.shape)
    data.to_csv("./datasets/merged_politifact.csv", index=False)


def initMergedPolitifact():
    data = pd.read_csv(
        "./datasets/merged_politifact.csv").reset_index(drop=True)
    data = applyToDF(data)  # from subjectivity
    data = cleanDataText(data)
    data["speaker"] = data["speaker"].str.lower()
    data["speaker"] = data["speaker"].str.replace(" ", "-", regex=False)
    data["speaker"] = data["speaker"].str.replace("'", "", regex=False)

    data = data[data["label"] != "full-flop"]
    data = data[data["label"] != "half-flip"]
    data = data[data["label"] != "no-flip"]
    data.drop_duplicates(subset="statement", keep='last', inplace=True)
    # 312 row with NaN
    # data.dropna(inplace=True)
    # print(data.isna().sum().sum())
    print(data)
    print(data.shape)

    print(data["label"].value_counts())
    print(data["subjectivity"].value_counts())

    data.to_csv("./cleanDatasets/clean_merged_politifact.csv", index=False)


def speaker2credit():
    s2c = pd.read_table(
        "./datasets/speaker2credit/s2c.tsv").reset_index(drop=True)
    s2c.rename(
        columns={"speakers_job": "speaker's job title", "state_info": "state", "party_affiliation": "party", "barely_true_cnt": "mostly false counts", "false_cnt": "false counts",  "half_true_cnt": "half true counts", "mostly_true_cnt": "mostly true counts", "pants_on_fire_cnt": "pants on fire counts", "true_cnt": "true counts"}, inplace=True)
    s2c.sort_values(by="speaker", inplace=True)
    s2c.to_csv(
        "./datasets/speaker2credit/s2c.csv", index=False)


def fillMissingMetadata():  # with speaker2credit
    data = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact.csv").reset_index(drop=True)
    s2c = pd.read_csv(
        "./datasets/speaker2credit/s2c.csv").reset_index(drop=True)
    data = pd.concat([pd.concat([data, s2c])])
    data.sort_values(by=["speaker", "speaker's job title",
                         "state", "party", "true counts", "mostly true counts",	"half true counts",	"mostly false counts", "false counts", "pants on fire counts"], inplace=True)
    data = data.groupby(["speaker"], as_index=False).apply(
        lambda group: group.ffill())
    data.dropna(subset=["statement"], inplace=True)
    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+s2c.csv", index=False)


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


def main():
    # mergePolitifact()
    # initMergedPolitifact()
    # speaker2credit()
    # fillMissingMetadata()
    # initFakeNewsNet()

    data = pd.read_csv(
        "./cleanDatasets/clean_merged_politifact+s2c_edited.csv").reset_index(drop=True)
    # data.sort_values(by=["speaker", "speaker's job title",
    #                      "state", "party", "true counts", "mostly true counts",	"half true counts",	"mostly false counts", "false counts", "pants on fire counts"], inplace=True)
    # data[["speaker's job title", "state", "party", "true counts", "mostly true counts", "half true counts",	"mostly false counts", "false counts", "pants on fire counts"]] = data[["speaker's job title", "state", "party", "true counts", "mostly true counts", "half true counts",	"mostly false counts", "false counts", "pants on fire counts"]].mask(
    #     data["speaker"].duplicated()).ffill()
    data[["speaker's job title", "state", "party", "true counts", "mostly true counts",
          "half true counts",	"mostly false counts", "false counts", "pants on fire counts"]].ffill()
    print(data[["speaker", "speaker's job title"]].iloc[19060:19080])
    # data = data.iloc[data.groupby('speaker').speaker.transform(
    #     'size').argsort(kind='mergesort')]
    print(data.isna().sum())
    data.to_csv(
        "./cleanDatasets/clean_merged_politifact+s2c_edited.csv", index=False)


main()

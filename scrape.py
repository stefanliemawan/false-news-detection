from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time
import re

# Scrape statements and metadata

labels = []
statements = []
subjects = []
speakers = []
sjts = []  # speaker"s job title
states = []
# parties = []
t_counts = []
mt_counts = []
ht_counts = []
mf_counts = []
f_counts = []
pf_counts = []
contexts = []
dates = []

sjt_patterns = [r"(\bU.S.+?\b)", r"(?<=\bis the \b)", r"(?<=\bis a \b)", r"(?<=\bis an \b)",
                r"(?<=\bserved as the \b)", r"(?<=\bhas served as \b)", r"(?<=\bhas served as a \b)", r"(?<=\bwas elected to the \b)", r"(?<=\bwas elected \b)", r"(?<=\bwas appointed as \b)", r"(?<=\bis serving as \b)", r"(?<=\bwas sworn in as \b)", r"(?<=\bis \b)"]

state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi",
               "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

politifact_url = "https://politifact.com/"
a_regex = r"href=[\"\'](.*?)[\"\']"


def scrape_speaker(url):
    webpage = requests.get(politifact_url + url)
    soup = BeautifulSoup(webpage.text, "html.parser")

    desc = soup.find("div", attrs={"class": "m-pageheader__body"})
    desc = desc.text.strip()

    for pattern in sjt_patterns:
        sjt = re.search(pattern+r"(?:[^\.]|\.(?=\d))*\.", desc)
        if sjt is not None:
            break
    try:
        sjt = sjt.group(0)
        sjt = sjt.title()
    except:
        sjt = None
    sjts.append(sjt)
    scorecard_checks = soup.find_all(
        "p", attrs={"class": "m-scorecard__checks"})
    checks = [i.text.strip().split(" ")[0] for i in scorecard_checks]
    t_counts.append(checks[0])
    mt_counts.append(checks[1])
    ht_counts.append(checks[2])
    mf_counts.append(checks[3])
    f_counts.append(checks[4])
    pf_counts.append(checks[5])


def scrape_statement(url):
    webpage = requests.get(politifact_url + url)
    soup = BeautifulSoup(webpage.text, "html.parser")

    statement_meta = soup.find(
        "div", attrs={"class": "m-statement__meta"})
    statement_desc = soup.find("div", attrs={"class": "m-statement__desc"})
    statement_quote = soup.find(
        "div", attrs={"class": "m-statement__quote"})
    statement_item = soup.find_all(
        "li", attrs={"class": "m-list__item"})
    statement_meter = soup.find("div", attrs={"class": "m-statement__meter"})

    speaker = statement_meta.find("a").text.strip()
    speakers.append(speaker)

    desc = statement_desc.text.strip().split()
    date = " ".join(desc[2:5])
    dates.append(date)

    context = desc[6:]
    if context:
        context[-1] = context[-1][:-1]
        context = " ".join(context)
        contexts.append(context)
    else:
        contexts.append(None)

    statement = statement_quote.text.strip()
    statements.append(statement)

    subject = [i.text.strip() for i in statement_item]
    if len(subject) != 0:
        subject.pop(-1)
    state = list(set(subject).intersection(state_names))
    try:
        state = state[0]
        subject.remove(state)
    except:
        state = None
    states.append(state)
    subject = ",".join(subject)
    subjects.append(subject)

    a_tags = [i.find('a', href=True) for i in statement_item]
    try:
        speaker_url = re.findall(
            a_regex, str(a_tags[-1]))[0]
    except:
        speaker_url = "personalities/" + \
            speaker.replace(" ", "-").lower()
    scrape_speaker(speaker_url)

    label = statement_meter.find(
        "div", attrs={"class": "c-image"}).find("img").get("alt")
    labels.append(label)


def scrape_website(page_number):
    page_num = str(page_number)
    URL = politifact_url + "/factchecks/list/?page=" + page_num
    webpage = requests.get(URL)
    soup = BeautifulSoup(webpage.text, "html.parser")

    div_tags = soup.find_all(
        "div", attrs={"class": "m-statement__quote"})
    for i in div_tags:
        a_tags = i.find_all("a", href=True)
        url = re.findall(
            a_regex, str(a_tags[0]))[0]
        scrape_statement(url)


def save(filename):
    data = pd.DataFrame(
        columns=["label", "statement", "subject", "speaker", "speaker's job title", "state", "party", "true counts", "mostly true counts", "half true counts", "mostly false counts", "false counts", "pants on fire counts", "context", "date"])
    data["label"] = labels
    data["statement"] = statements
    data["subject"] = subjects
    data["speaker"] = speakers
    data["speaker's job title"] = sjts
    data["state"] = states
    # data["party"] = parties
    data["true counts"] = t_counts
    data["mostly true counts"] = mt_counts
    data["half true counts"] = ht_counts
    data["mostly false counts"] = mf_counts
    data["false counts"] = f_counts
    data["pants on fire counts"] = pf_counts
    data["context"] = contexts
    data["date"] = dates

    print(data)
    print(data.shape)
    print("Data saved to %s.csv" % filename)

    data.to_csv("datasets/PolitiFact/%s.csv" % filename, index=False)


start = time.time()

n = 648
first_page = 1
for i in range(first_page, n+1):
    print("Processing Page ", i, "...")
    scrape_website(i)
    if i % 20 == 0:
        save(str(first_page)+"-"+str(i))


save("648pages_politifact_full_metadata")
# save("test")


end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("######################################")
print("{:0>2} hours  {:0>2} minutes  {:05.2f} seconds".format(
    int(hours), int(minutes), seconds))
print("######################################")

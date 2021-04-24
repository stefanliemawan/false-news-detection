from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time
import re


politifact_url = "https://politifact.com/"
a_regex = r"href=[\"\'](.*?)[\"\']"

speakers = []
parties = []
t_counts = []
mt_counts = []
ht_counts = []
mf_counts = []
f_counts = []
pf_counts = []


def scrape_profile(url):
    webpage = requests.get(politifact_url+url)
    soup = BeautifulSoup(webpage.text, "html.parser")

    scorecard_checks = soup.find_all(
        "p", attrs={"class": "m-scorecard__checks"})
    checks = [i.text.strip().split(" ")[0] for i in scorecard_checks]
    t_counts.append(checks[0])
    mt_counts.append(checks[1])
    ht_counts.append(checks[2])
    mf_counts.append(checks[3])
    f_counts.append(checks[4])
    pf_counts.append(checks[5])


def scrape_party():
    webpage = requests.get(politifact_url+"personalities/")
    soup = BeautifulSoup(webpage.text, "html.parser")

    profiles = soup.find_all("div", attrs={"class": "c-chyron"})

    for profile in profiles:
        p = profile.text.strip().split("\n")
        speaker = p[0]
        speakers.append(speaker)
        print("Scraping", speaker)
        party = p[-1]
        parties.append(party)

        a_tags = profile.find('a', href=True)
        speaker_url = re.findall(
            a_regex, str(a_tags))[0]
        scrape_profile(speaker_url)


def main():
    scrape_party()
    data = pd.DataFrame(columns=["speaker", "party",  "true counts", "mostly true counts",
                                 "half true counts", "mostly false counts", "false counts", "pants on fire counts"])
    data["speaker"] = speakers
    data["party"] = parties
    data["true counts"] = t_counts
    data["mostly true counts"] = mt_counts
    data["half true counts"] = ht_counts
    data["mostly false counts"] = mf_counts
    data["false counts"] = f_counts
    data["pants on fire counts"] = pf_counts

    print(data)
    print(data.shape)

    data.to_csv("datasets/PolitiFact/profiles.csv", index=False)


main()

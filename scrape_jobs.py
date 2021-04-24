from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time
import re

wiki_url = "https://en.wikipedia.org/wiki/"
# "Alan_Hays"
# th colspan="2" class="infobox-header"


def scrape_jobs(speaker):
    speaker_url = speaker.replace(" ", "_")
    webpage = requests.get(wiki_url+speaker_url)
    soup = BeautifulSoup(webpage.text, "html.parser")
    tags = soup.find("th", attrs={"class": "infobox-header"})
    job = None
    if tags is not None:
        job = tags.text.strip()
        job = job.replace("from", " from")
        print(speaker_url, job)
        data.to_csv("datasets/PolitiFact/profiles_wikipedia.csv", index=False)
    return job


data = pd.read_csv(
    "datasets/PolitiFact/profiles - Copy.csv").reset_index(drop=True)
data = data.sort_values(by=["speaker"]).reset_index(drop=True)

data["speaker's job title"] = data["speaker"].apply(scrape_jobs)

print(data["speaker's job title"].isna().sum())

data.to_csv("datasets/PolitiFact/profiles_wikipedia.csv", index=False)

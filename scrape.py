# based on https://randerson112358.medium.com/scrape-a-political-website-for-fake-real-news-using-python-b4f5b2af830b
from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time

speakers = []
dates = []
platforms = []
checkers = []
statements = []
labels = []


def scrape_website(page_number):
    page_num = str(page_number)
    URL = 'https://www.politifact.com/factchecks/list/?page=' + \
        page_num
    webpage = requests.get(URL)
    soup = BeautifulSoup(webpage.text, "html.parser")

    statement_meta = soup.find_all(
        'div', attrs={'class': 'm-statement__meta'})
    statement_desc = soup.find_all('div', attrs={'class': 'm-statement__desc'})
    statement_quote = soup.find_all(
        'div', attrs={'class': 'm-statement__quote'})
    statement_footer = soup.find_all(
        'footer', attrs={'class': 'm-statement__footer'})
    label = soup.find_all('div', attrs={'class': 'm-statement__meter'})

    for i in statement_meta:
        link_meta = i.find_all('a')  # Source
        source_text = link_meta[0].text.strip()
        speakers.append(source_text)

    for i in statement_desc:
        link_desc = i.text.strip()
        desc = link_desc.split()

        month = desc[2]
        day = desc[3]
        year = desc[4]
        date = month+' '+day+' '+year
        dates.append(date)

        platform = desc[6:]
        if platform:
            platform[-1] = platform[-1][:-1]
        platform = " ".join(platform)
        platforms.append(platform)

    for i in statement_footer:
        link_footer = i.text.strip()
        name_and_date = link_footer.split()
        first_name = name_and_date[1]
        last_name = name_and_date[2]
        checker = first_name+' '+last_name
        checkers.append(checker)

    for i in statement_quote:
        link_quote = i.find_all('a')
        statement = link_quote[0].text.strip()
        statements.append(statement)

    for i in label:
        fact = i.find('div', attrs={'class': 'c-image'}).find('img').get('alt')
        labels.append(fact)


def save(filename):
    data = pd.DataFrame(
        columns=['checker',  'statement', 'speaker', 'date', 'label'])
    data['checker'] = checkers
    data['statement'] = statements
    data['speaker'] = speakers
    data['date'] = dates
    data['label'] = labels

    print(data)
    print("Data saved to %s.csv" % filename)

    data.to_csv('datasets/PolitiFact/%s.csv' % filename)


start = time.time()

n = 641
for i in range(1, n+1):
    print("Processing Page ", i)
    scrape_website(i)
    if i % 100 == 0:
        save("temp")

save("641pages")


end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("######################################")
print("{:0>2} hours  {:0>2} minutes  {:05.2f} seconds".format(
    int(hours), int(minutes), seconds))
print("######################################")

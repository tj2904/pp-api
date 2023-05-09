import sentry_sdk
import os
import urllib.request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.sentiment.vader
from deta import Deta
import feedparser
nltk.download('vader_lexicon', download_dir='/tmp')
nltk.data.path.append('/tmp')

sentry_sdk.init(
    dsn="https://5a0e51d4d9df41cf963941e56b6f71d6@o4505121660665856.ingest.sentry.io/4505127392837632",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

detaBaseApiKey = os.getenv("Deta-Base")

deta = Deta(detaBaseApiKey)
dbBasicVader = deta.Base("basicVaderScoredNews")

def cron_task():
    bbc_feed_new = feedparser.parse(
        "http://feeds.bbci.co.uk/news/england/rss.xml")
    items = bbc_feed_new.entries

    for item in items:
        title = item.title
        summary = item.summary
        id = item.id
        published_parsed = item.published

        response = urllib.request.urlopen(id)
        soup = BeautifulSoup(response, 'html.parser',
                         from_encoding=response.info().get_param('charset'))
        image_url = soup.find("meta", property="og:image")["content"]
        image = image_url

        dbBasicVader.insert({
            "title": title,
            "summary": summary,
            "id": id,
            "published": published_parsed,
            "source": "bbc",
            "region": "england",
            "imageUrl": image,
        })

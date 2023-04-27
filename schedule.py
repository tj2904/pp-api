import os
import urllib.request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.sentiment.vader
from deta import Deta
from fastapi import FastAPI, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import feedparser
import pandas as pd
nltk.download('vader_lexicon', download_dir='/tmp')
nltk.data.path.append('/tmp')

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

        db.insert({
            "title": title,
            "summary": summary,
            "id": id,
            "published": published_parsed,
            "source": "bbc",
            "region": "england",
            "imageUrl": image,
        })

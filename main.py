# to activate the virtual environment, run the following command in the terminal
# source venv/bin/activate

# to run the server, run the following command in the terminal
# uvicorn main:app --reload

import sentry_sdk
import os
from typing import List, Union
from fastapi import FastAPI, status, Request
from pydantic import BaseModel, HttpUrl
from deta import Deta
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.sentiment.vader
nltk.download('vader_lexicon')

import urllib.request
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()
detaBaseApiKey = os.getenv("DetaBase")

sentry_sdk.init(
    dsn="https://5a0e51d4d9df41cf963941e56b6f71d6@o4505121660665856.ingest.sentry.io/4505127392837632",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

class Vader(BaseModel):
    neg: float
    neu: float
    pos: float
    compound: float


class NewsResponse(BaseModel):
    title: str
    summary: str
    vaderTitle: Vader
    vaderSummary: Vader
    id: str
    imageUrl: HttpUrl = None
    published: List[int] = None


class Url(BaseModel):
    url: HttpUrl


class UrlResponse(BaseModel):
    image: HttpUrl


class HealthCheck(BaseModel):
    healthcheck: str


tags_metadata = [
    {
        "name": "default"
    },
    {
        "name": "Vader",
        "description": "Interaction with the VADER Sentiment Analysis Algorithm."
    },
    {
        "name": "Utilities",
        "description": "Utility endpoints for the API."
    },
]
app = FastAPI(title="PositivePress",
              description="Supporting APIs", openapi_tags=tags_metadata, version="0.1.0")

deta = Deta(detaBaseApiKey)
dbBasicVader = deta.Base("basicVaderScoredNews")

@app.get('/api/healthcheck', response_model=HealthCheck, status_code=status.HTTP_200_OK)
def perform_healthcheck():
    return {'healthcheck': 'Everything OK!'}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/v1/vader/live/england", tags=["Vader"])
def vader_scores_appended_to_BBC_England_news_feed():

    bbc_feed_new = feedparser.parse(
        "http://feeds.bbci.co.uk/news/england/rss.xml")
    items = bbc_feed_new.entries

    df = pd.DataFrame(columns=['title', 'summary',
                      'vaderTitle', 'vaderSummary'])

    for item in items:
        title = item.title
        summary = item.summary

        sid = SentimentIntensityAnalyzer()
        ss_title = sid.polarity_scores(title)
        ss_sumamry = sid.polarity_scores(summary)
        vader_title = ss_title
        vader_summary = ss_sumamry

        id = item.id
        published_parsed = item.published_parsed

        response = urllib.request.urlopen(id)
        soup = BeautifulSoup(response, 'html.parser',
                             from_encoding=response.info().get_param('charset'))
        image_url = soup.find("meta", property="og:image")["content"]
        image = image_url

        row = {'title': title, 'summary': summary, 'vaderTitle': vader_title,
               'vaderSummary': vader_summary, 'id': id, 'imageUrl': image, 'published': published_parsed}
        df = df.append(row, ignore_index=True)

    return df.to_dict(orient="records")


@app.get("/api/v1/vader/live/tech", tags=["Vader"])
def vader_scores_appended_to_BBC_tech_news_feed():

    bbc_feed_new = feedparser.parse(
        "http://feeds.bbci.co.uk/news/technology/rss.xml")
    items = bbc_feed_new.entries

    df = pd.DataFrame(columns=['title', 'summary',
                      'vaderTitle', 'vaderSummary'])

    for item in items:
        title = item.title
        summary = item.summary

        sid = SentimentIntensityAnalyzer()
        ss_title = sid.polarity_scores(title)
        ss_sumamry = sid.polarity_scores(summary)
        vader_title = ss_title
        vader_summary = ss_sumamry

        id = item.id
        published_parsed = item.published_parsed

        response = urllib.request.urlopen(id)
        soup = BeautifulSoup(response, 'html.parser',
                             from_encoding=response.info().get_param('charset'))
        image_url = soup.find("meta", property="og:image")["content"]
        image = image_url

        row = {'title': title, 'summary': summary, 'vaderTitle': vader_title,
               'vaderSummary': vader_summary, 'id': id, 'imageUrl': image, 'published': published_parsed}
        df = df.append(row, ignore_index=True)

    return df.to_dict(orient="records")


@app.get("/api/v1/vader/live/{category}", tags=["Vader"])
def vader_scores_appended_to_given_BBC_news_feed(category: str):

    category = category.lower()
    bbc_feed_new = feedparser.parse(
        'http://feeds.bbci.co.uk/news/'+category+'/rss.xml')
    items = bbc_feed_new.entries

    df = pd.DataFrame(columns=['title', 'summary',
                      'vaderTitle', 'vaderSummary'])

    for item in items:
        title = item.title
        summary = item.summary

        sid = SentimentIntensityAnalyzer()
        ss_title = sid.polarity_scores(title)
        ss_sumamry = sid.polarity_scores(summary)
        vader_title = ss_title
        vader_summary = ss_sumamry

        id = item.id
        published_parsed = item.published_parsed

        response = urllib.request.urlopen(id)
        soup = BeautifulSoup(response, 'html.parser',
                             from_encoding=response.info().get_param('charset'))
        image_url = soup.find("meta", property="og:image")["content"]
        image = image_url

        row = {'title': title, 'summary': summary, 'vaderTitle': vader_title,
               'vaderSummary': vader_summary, 'id': id, 'imageUrl': image, 'published': published_parsed}
        df = df.append(row, ignore_index=True)

    return df.to_dict(orient="records")

# get the highest scoring news story by summary compound
# {"vaderSummary.compound?gt": 0.75}
@app.get("/api/v1/vader/summary/pos/top", tags=["Vader"])
async def get_most_positive_vader_scored_news_from_database():
    result = dbBasicVader.fetch({"vaderSummary.compound?gt": 0.75})
    return {"data": result} if result else ({"message": "No news found"})

# scrape OpenGraph tags to provide an image for a given news url
@app.post("/api/v1/og/", response_model=UrlResponse, tags=["Utilities"])
def get_open_graph_image(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser',
                         from_encoding=response.info().get_param('charset'))
    image = soup.find("meta", property="og:image")["content"]
    return {'image': image}


@app.get("/api/v1/vader/store/england", tags=["Vader"])
def vader_bbc_england_news_to_database():
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

        sid = SentimentIntensityAnalyzer()
        ss_title = sid.polarity_scores(title)
        ss_summary = sid.polarity_scores(summary)
        vader_title = ss_title
        vader_summary = ss_summary

        dbBasicVader.insert({
            "title": title,
            "summary": summary,
            "id": id,
            "imageUrl": image,
            "published": published_parsed,
            "source": "bbc",
            "region": "england",
            "vaderTitle": vader_title,
            "vaderSummary": vader_summary
        })

    return {"message": "successful"}


@app.get("/api/v1/vader/score/{text}", tags=["Vader"])
async def vader_score_supplied_text(text: str):
    sid = SentimentIntensityAnalyzer()
    scored_text = sid.polarity_scores(text)
    return {"data": scored_text} if scored_text else ({"error": "Bad request"}, 400)


@app.get("/api/v1/vader/all", tags=["Vader"])
async def get_all_vader_scored_news_from_database():
    res = dbBasicVader.fetch([{"vaderSummary.compound?gte": 0.5}, {
        "vaderTitle.compound?gte": 0.5}])
    return {"data": res} if res else ({"message": "No news found"})

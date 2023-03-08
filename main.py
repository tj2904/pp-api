# to activate the virtual environment, run the following command in the terminal
# source venv/bin/activate

# to run the server, run the following command in the terminal
# uvicorn main:app --reload

from typing import List, Union
from fastapi import FastAPI, status, Request
from pydantic import BaseModel, HttpUrl
import feedparser
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.sentiment.vader

import urllib.request
from urllib.parse import urlparse


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
        "name": "Utilities",
        "description": "Utility endpoints for the API."
    },
]

app = FastAPI(title="PositivePress",
              description="Supporting APIs", openapi_tags=tags_metadata, version="0.1.0")


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


@app.post("/api/v1/vader/live/{category}", tags=["Vader"])
def vader_scores_appended_to_given_BBC_news_feed(category: str):

    bbc_feed_new = feedparser.parse(
        'http://feeds.bbci.co.uk/news/{category}/rss.xml')
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



# scrape OpenGraph tags to provide an image for a given news url
@app.post("/api/v1/og/", response_model=UrlResponse, tags=["Utilities"])
def get_open_graph_image(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser',
                         from_encoding=response.info().get_param('charset'))
    image = soup.find("meta", property="og:image")["content"]
    return {'image': image}

#!/usr/bin/env python
import feedparser
from scrapetools import dic2csv

FILE_PATH = "RawData/RSS_Data"
SOURCES_PATH = "sources"

data = []
sources = []

with open(SOURCES_PATH, "r") as infile:
    for line in infile:
        sources.append(line)

for source in sources:
    feed = feedparser.parse(source)
    for entry in feed.entries:
        info = {"headline": entry.title,
            "source": feed['feed']['title'],
            "time": entry.published_parsed,
            "link": entry.link}
        data.append(info)

dic2csv(data, FILE_PATH)

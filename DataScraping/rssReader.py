import sys
import feedparser
import csv
from scrapeTools import dic2csv
data = []
sources = []

with open("sources", "r") as infile:
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
dic2csv(data, "RawData/RSS_Data2")

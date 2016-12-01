import sys
import feedparser
import csv

data = []

def dic2csv(dicti):
        with open('rssInfo3.csv', 'w') as f:
                w = csv.DictWriter(f,dicti[0].keys())
                w.writeheader()
                w.writerows(dicti)
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
dic2csv(data)

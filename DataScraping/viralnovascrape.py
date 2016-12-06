#!/usr/bin/env python
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from scrapetools import dic2csv
import sys

PAGE_NUMBER = 600
FILE_PATH = "RawData/viralNova.txt"
BASE_URL = "http://www.viralnova.com/"

def get_headlines(link, depth, list):
    if(depth >= PAGE_NUMBER):
        return list
    hdr = {'User-Agent': 'Mozilla/5.0'}
    request = Request(BASE_URL, headers = hdr)
    html = urlopen(request).read()
    soup = BeautifulSoup(html, "lxml")
    list.extend(soup.findAll("h4"))
    nextLink = soup.find(class_="loadmore").a["href"]
    return get_headlines(nextLink, depth+1, list)

def write_file(list):
    headlines = []
    for x in list:
        if len(x) == 1: 
            headlines.append(x.contents[0])
    with open(FILE_PATH, "w") as f:
        for title in headlines:
            f.write("%s\n" % title)
    print(len(headlines))

h4 = get_headlines(BASE_URL, 1, [])
write_file(h4)

import numpy as np #numpy for calculations and certain mathematical arrays
import pandas as pd #pandas for dataframes and importing data
import zstandard as zstd #zstandard lets us read condensed data | works with pandas
import sklearn as sk #cosine similarity for mass calculation - (super fast)
import operator as op #should help with iterators | should be used in most popular and trending data sets


def read_zstd(path: str): #helps read condensed zst files
    with open(path, "rb") as f:
        return pd.read_csv(zstd.ZstdDecompressor().stream_reader(f))

testo = open('test.txt', 'w')

uservalues = pd.read_csv('profiles.csv') #our actual user profiles

start = pd.read_csv('start.csv') #acts as our beginning set for identifying shows

indepth = read_zstd("anime.csv.zst") #has master list / not all are used as some belong to other services

shows = pd.read_csv('animes.csv') #all shows and their ratings w/ genre


print(uservalues)
print(indepth)
print(shows)

idmap = {} #our user data contains ids instead of show names, our data just uses names

for row in start.itertuples(index=True): #generating a map from starts ids to title for later crossreference
    if (getattr(row, "uid")) not in idmap:
        idmap[getattr(row, "uid")] = getattr(row, "title")
    else: #there are repeat ids, this just makes the ids map to the base title in case of id:show -> id:show season 2
        if idmap.get(getattr(row, "uid")) in getattr(row, "title"):
            idmap[getattr(row, "uid")] = getattr(row, "title")

showmap = {}

for show in shows.itertuples(index=True): #first step of making shows to compare, vector can be used for cosine similarity but not done yet
    vector = []
    for i in range(13, 41):
        vector.append(show[i])
    showmap[show[1]] = (show[2], show[3], show[7], vector)

print(len(idmap))
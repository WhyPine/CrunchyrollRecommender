import numpy as np #numpy for calculations and certain mathematical arrays
import pandas as pd #pandas for dataframes and importing data
import zstandard as zstd #zstandard lets us read condensed data | works with pandas
import sklearn as sk #cosine similarity for mass calculation - (super fast)
import operator as op #should help with iterators | should be used in most popular and trending data sets
from sklearn.metrics.pairwise import cosine_similarity
# SVD ML recommendation library
from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection.split import train_test_split
from surprise.accuracy import rmse

def read_zstd(path: str): #helps read condensed zst files
    with open(path, "rb") as f:
        return pd.read_csv(zstd.ZstdDecompressor().stream_reader(f))

top100 = []

testo = open('test.txt', 'w')

uservalues = pd.read_csv('profiles.csv') #our actual user profiles

start = pd.read_csv('start.csv') #acts as our beginning set for identifying shows

indepth = read_zstd("anime.csv.zst") #has master list / not all are used as some belong to other services

shows = pd.read_csv('animes.csv') #all shows and their ratings w/ genre

reviews = pd.read_csv("reviews.csv")

print(reviews)
print(uservalues)
print(indepth)
print(shows)

animeIdToName = {} #our user data contains ids instead of show names, our data just uses names

for row in start.itertuples(index=True): #generating a map from starts ids to title for later crossreference
    if (getattr(row, "uid")) not in animeIdToName:
        animeIdToName[getattr(row, "uid")] = getattr(row, "title")
    else: #there are repeat ids, this just makes the ids map to the base title in case of id:show -> id:show season 2
        if animeIdToName.get(getattr(row, "uid")) in getattr(row, "title"):
            animeIdToName[getattr(row, "uid")] = getattr(row, "title")

showmap = {}

for show in shows.itertuples(index=True): #first step of making shows to compare, vector can be used for cosine similarity but not done yet
    genres = np.zeros(28, dtype='int16')
    for i in range(13, 41):
        if show[i]:
            genres[i-13] = 1
    #       title         url    weight    rate    genre vectord
    showmap[show[1]] = (show[2], show[6], show[7], genres)

print(len(animeIdToName))
print(animeIdToName)

print(len(showmap))

def popular():
    v = list(showmap.values())
    v.sort(key=lambda a: (a[1] * a[2]))
    k = list(showmap.keys())
    for i in range(0,100):
        temp = v[len(v) - (1 + i)]
        print(temp)
        top100.append(temp)

popular()

# will remain in order
shownames = list(showmap.keys())
showgenres = []
# getting array of genre list
for value in showmap.values():
    showgenres.append(value[3])
showgenres = np.array(showgenres)
# generating similarity matrix
genresimilaritymatrix = cosine_similarity(showgenres)

drSTONEsimilarshows = []
for x in range(1255):
    drSTONEsimilarshows.append(x)
# sorting all shows by similarity to dr. STONE genres
drSTONEsimilarshows.sort(key=lambda x: genresimilaritymatrix[128][x], reverse=True)
for x in range(10):
    print(shownames[drSTONEsimilarshows[x]], "similarity =", genresimilaritymatrix[128][drSTONEsimilarshows[x]],  drSTONEsimilarshows[x])
# genresimilaritymatrix[128]


# [index, movieid, rating [1-5]]
# movieRatings = list()
# for index, row in uservalues.iterrows():
#     list = row[3].replace("[", "").replace("]", "").replace("'", "").replace (" ", "")
#     for movie in list.split(','):
#         try:
#             movieId = int(movie)
#         except:
#             continue
#         new_entry = pd.DataFrame(columns = ["userId", "movieId", "Rating"])
#         new_entry.loc[0] = [index, movieId, 5]
#         movieRatings.append(new_entry)
#     if index % 100 == 0:
#         print("index =", index)
# movieRatings = pd.concat(movieRatings)
# print(movieRatings)

# movieRatings.to_csv("./transformedfavorites.csv", sep='\t')
movieRatings = pd.read_csv("./transformedfavorites.csv", sep='\t')

# reader = Reader(rating_scale=(1,5))
# data = Dataset.load_from_df(movieRatings[["userId", "movieId", "Rating"]], reader)
# data = list()
# for index, row in reviews.iterrows():
#     new_entry = pd.DataFrame(columns = ["userId", "animeId", "Rating"])
#     new_entry.loc[0] = [int(row[0]), int(row[2]), int(row[4])]
#     data.append(new_entry)
#     if index%25000 == 0:
#         print(index)
# data = pd.concat(data)

def getTopPredictions(predictions, n=10):
    print("in predictions")
    # Map predictions to each user
    result = {}
    for userId, animeId, rating, estimate, _ in predictions:
        result[userId].append((animeId, estimate))
    print("mapped predictions to users")
    # Now grab top n predictions for each user
    for userId, estimates in result.items():
        estimates.sort(key=lambda x: x[1], reverse = True)
        result[userId] = estimates[:n]
        print("sorted user", userId)


reader = Reader(rating_scale=(1,10))
reviewsData = Dataset.load_from_df(reviews[["profile", "anime_uid", "score"]], reader)

trainSet = reviewsData.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

testSet = trainSet.build_testset()
predictions = algo.test(testSet)
rmse(predictions)

UserToAnimePredictions = {}
length = len(uservalues.index)
for row in uservalues.itertuples():
    if(row[0]%100 == 0):
        print(row[0]/length)
    UserToAnimePredictions[row[1]] = []
    # row[0] is profile name
    for animeId in animeIdToName.keys():
        prediction = algo.predict(row[1], animeId, r_ui=None, verbose=False)
        UserToAnimePredictions[row[1]].append((prediction[1], prediction[3]))
    if(row[0] == 500):
        break

print(UserToAnimePredictions["DesolatePsyche"])
# x = algo.predict("DesolatePsyche", 38816, r_ui=None, verbose=False) returns tuple [profile, animeID, original value, estimated rating]


# for anime in animeIdToName.keys():
#     for user in userIdToName.keys():
#         algo.predict(user,anime, r_ui=None,verbose = true)

# The code below takes ages to run :(
# testSet = trainSet.build_anti_testset()
# print("built anti-test set")
# predictions = algo.test(testSet)
# print("predictions complete", predictions)
# recommendations = getTopPredictions(predictions, n=10)
# print(recommendations)


algo = SVD()
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(movieRatings[["userId", "movieId", "Rating"]], reader)
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)



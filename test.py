# Basic data handing libraries
import numpy as np #numpy for calculations and certain mathematical arrays
import pandas as pd #pandas for dataframes and importing data
import zstandard as zstd #zstandard lets us read condensed data | works with pandas
from sklearn.metrics.pairwise import cosine_similarity

# SVD ML recommendation libraries
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.accuracy import rmse

# TensorFlow ML recommendation libraries
import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras

# Nice to haves
from datetime import datetime
import time
import json

start_time = time.time()

def read_zstd(path: str): #helps read condensed zst files
    with open(path, "rb") as f:
        return pd.read_csv(zstd.ZstdDecompressor().stream_reader(f))

# uservalues = pd.read_csv('profiles.csv') #our actual user profiles

start = pd.read_csv('start.csv') #acts as our beginning set for identifying shows

# indepth = read_zstd("anime.csv.zst") #has master list / not all are used as some belong to other services

shows = pd.read_csv('animes.csv') #all shows and their ratings w/ genre

reviews = pd.read_csv("reviews.csv")

movieRatings = pd.read_csv("./transformedfavorites.csv", sep='\t')

# print(reviews)
# print(uservalues)
# print(indepth)
# print(shows)

# Setting up reccomendation samples
ReccomendationSamples = {}
Usernames = ["DesolatePsyche", "Plasmatize", "ApertureEmployee", "AnthraxHierarchy", "TheAfroNinja", "EpicSawce", "TheAnimeGandalf", "CoolBreeze", "TheFifthRider", "AStupidPotato"]
for name in Usernames:
    ReccomendationSamples[name] = []


animeIdToName = {} # Mapping show id's to show names

# generating a map from starts ids to title for later crossreference
for row in start.itertuples(index=True): 
    if (getattr(row, "uid")) not in animeIdToName:
        animeIdToName[getattr(row, "uid")] = getattr(row, "title")
    else: #there are repeat ids, this just makes the ids map to the base title in case of id:show -> id:show season 2
        if animeIdToName.get(getattr(row, "uid")) in getattr(row, "title"):
            animeIdToName[getattr(row, "uid")] = getattr(row, "title")

showmap = {}

# Generating dictionary mapping show id to data + building genre vector
for show in shows.itertuples(index=True): 
    genres = np.zeros(28, dtype='int16')
    for i in range(13, 41):
        if show[i]:
            genres[i-13] = 1
    #       title         url    votes     rate    genre vector
    showmap[show[1]] = (show[2], show[5], show[7], genres)
read_time = time.time()

# [0] Top 100 most popular shows based on user ratings
def popular():
    v = list(showmap.keys())
    v.sort(key=lambda x: (showmap[x][1] * showmap[x][2]), reverse=True)
    return v[:100]
print(popular())
zero_time = time.time()

# [1] Basic Item-Item reccomender using cosine similarity and genre data
shownames = list(showmap.keys())
showgenres = []
# getting array of genre list
for value in showmap.values():
    showgenres.append(value[3])
showgenres = np.array(showgenres)
# generating similarity matrix
genresimilaritymatrix = cosine_similarity(showgenres)
one_time = time.time()

# drSTONEsimilarshows = []
# for x in range(1255):
#     drSTONEsimilarshows.append(x)
# # sorting all shows by similarity to dr. STONE genres
# drSTONEsimilarshows.sort(key=lambda x: genresimilaritymatrix[128][x], reverse=True)
# for x in range(10):
#     print(shownames[drSTONEsimilarshows[x]], "similarity =", genresimilaritymatrix[128][drSTONEsimilarshows[x]],  drSTONEsimilarshows[x])


# [2] User-User ML reccomender using SVD and thousands of user reviews

# Setting up data
reader = Reader(rating_scale=(1,10))
reviewsData = Dataset.load_from_df(reviews[["profile", "anime_uid", "score"]], reader)

# Building trainset will all available data
trainSet = reviewsData.build_full_trainset()
algo1 = SVD()

# Running the algorithm
algo1.fit(trainSet)
testSet = trainSet.build_testset()
predictions = algo1.test(testSet)
rmse(predictions)

# Getting reccomendations from model
for user in Usernames:
    UserReccomendations = []
    for animeId in animeIdToName.keys():
         prediction = algo1.predict(user, animeId, r_ui=None, verbose=False)
                               #    animeID        estimated rating
         UserReccomendations.append((prediction[1], prediction[3]))
    UserReccomendations.sort(key=lambda x: x[1], reverse = True)
    UserReccomendations = UserReccomendations[:20]
    ReccomendationSamples[user].append(UserReccomendations)
two_time = time.time()

# [3] User-User ML reccomender using SVD and only user favorites

# Setting up algorithm and data 
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(movieRatings[["userId", "movieId", "Rating"]], reader)

# Training this time using cross validation
algo2 = SVD()
cross_validate(algo2, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# Getting reccomendations from model
for user in Usernames:
    UserReccomendations = []
    for animeId in animeIdToName.keys():
         prediction = algo2.predict(user, animeId, r_ui=None, verbose=False)
                               #    animeID        estimated rating
         UserReccomendations.append((prediction[1], prediction[3]))
    UserReccomendations.sort(key=lambda x: x[1], reverse = True)
    UserReccomendations = UserReccomendations[:20]
    ReccomendationSamples[user].append(UserReccomendations)
three_time = time.time()

# [4] User-User ML reccommender using TensorFlow and thousands of user reviews

# Formatting adjustments for compatibility
reviewsData = reviews[["profile", "anime_uid", "score"]]
reviewsData['profile'] = reviewsData.profile.astype(np.str)
reviewsData['anime_uid'] = reviewsData.anime_uid.astype(np.str)
reviewsData['score'] = reviewsData.score.astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices((tf.cast(reviewsData['profile'].values.reshape(-1,1), tf.string),    tf.cast(reviewsData['anime_uid'].values.reshape(-1,1), tf.string),
tf.cast(reviewsData['score'].values.reshape(-1,1),tf.float32)))

# Preparing data for tensorflow
@tf.function
def rename(x0,x1,x2):
    y = {}
    y["profile"] = x0
    y['anime_uid'] = x1
    y['score'] = x2
    return y
dataset = dataset.map(rename)
animes = reviewsData.anime_uid.values
users = reviewsData.profile.values
unique_anime_titles = np.unique(list(animes))
unique_user_ids = np.unique(list(users))

# Model definitions
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for animes.
    self.anime_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_anime_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_anime_titles) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def __call__(self, x):
    
    profile, anime_uid = x
    user_embedding = self.user_embeddings(profile)
    anime_embedding = self.anime_embeddings(anime_uid)

    return self.ratings(tf.concat([user_embedding, anime_embedding], axis=1))
  
class AnimeModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss = tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features, training=False) -> tf.Tensor:
        print(features)
        rating_predictions = self.ranking_model((features['profile'], features["anime_uid"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["score"], predictions=rating_predictions)

# Instantiating AnimeModel & preparing for training
model = AnimeModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
cache_dataset = dataset.cache()
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Training, increasing epochs increases computation cost and accuracy
model.fit(cache_dataset, epochs=1, verbose=1, callbacks=[tensorboard_callback])

# Formatting function
@tf.function
def rename_test(x0,x1):
    y = {}
    y["profile"] = x0
    y['anime_uid'] = x1
    return y

# Getting reccomendations from model
for user in Usernames:
    UserReccomendations = []
    userArray = np.array([user for i in range(len(unique_anime_titles))])
    testData = tf.data.Dataset.from_tensor_slices((tf.cast(userArray.reshape(-1,1), tf.string), tf.cast(unique_anime_titles.reshape(-1,1), tf.string)))
    testData = testData.map(rename_test)
    test_ratings = {}
    for b in testData:
        test_ratings[b['anime_uid'].numpy()[0]] = model.ranking_model((b['profile'],b['anime_uid']))
    for b in sorted(test_ratings, key=test_ratings.get, reverse=True)[:20]:
        UserReccomendations.append((int(tf.compat.as_str_any(b)), float(tf.get_static_value(test_ratings.get(b)))))
    ReccomendationSamples[user].append(UserReccomendations)
four_time = time.time()

with open("test.json", "w") as file:
    json.dump(ReccomendationSamples, file, indent=4)


print("Execution Times:")
print("Reading the data:", int(read_time - start_time), "seconds")
print("[0] Top 100 Item-Item           :", int(zero_time - read_time), "seconds")
print("[1] Genre Based Item-Item       :", int(one_time - zero_time), "seconds")
print("[2] User-User SVD Review-based  :", int(two_time - one_time), "seconds")
print("[3] User-User SVD favorite-based:", int(three_time - two_time), "seconds")
print("[4] User-User TF Review-based   :", int(four_time - three_time), "seconds")
end_time = time.time()
print("Total Execution Time:", int(end_time - start_time), "seconds")
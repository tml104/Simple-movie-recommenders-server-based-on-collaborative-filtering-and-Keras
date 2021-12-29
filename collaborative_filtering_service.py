import imp
from logging import root
from typing import List
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from pathlib import Path

from aiohttp import web
import logging as lg
import json

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


#Load csv
root_path = Path(".")
ratings_file = Path(root_path / "ratings.csv")
movie_df = spark.read.csv(str(root_path / "movies2.csv"),header=True,sep="::",encoding="UTF-8")
df = spark.read.csv(str(ratings_file),header=True)


#preprocess
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

model = keras.models.load_model("./CF_model")
lg.info("Load model succeeded.")


def work():
    app = web.Application()
    routes = web.RouteTableDef()

    @routes.post('/')
    async def post_handler(request: web.Request):
        j = await request.json()
        movies_watched_by_user : List = j["movieId"]
        movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user)]["movieId"]
        movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
        movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

        user_id = 6042 #not exist
        user_encoder = user2user_encoded.get(user_id,user_id-1)
        user_movie_array = np.hstack(
            ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
        )
        user_movie_array = tf.cast(user_movie_array,dtype=tf.int64)
        print(user_movie_array)
        ratings = model.predict(user_movie_array).flatten()
        top_ratings_indices = ratings.argsort()[-30:][::-1]
        recommended_movie_ids = [
            movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
        ]

        ret_response = {
            "recommended_movie_ids": recommended_movie_ids
        }
        return web.json_response(ret_response)
    
    app.add_routes(routes)
    web.run_app(app)


if __name__ == "__main__":
    work()
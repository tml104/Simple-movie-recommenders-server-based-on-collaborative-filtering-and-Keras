import requests
import pandas as pd


def run_test():

    payload = {
        "movieId" : [1340,1341,1345,1346,1347,1348,1350,1356,1358,2301,2302]
    }
    r = requests.post("http://localhost:8080/",json=payload)
    r = r.json()
    print(r)

    movie_df = pd.read_csv("./movies2.csv", delimiter="::")
    print(movie_df[movie_df["movieId"].isin(r['recommended_movie_ids'])])


if __name__ == "__main__":
    run_test()
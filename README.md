# Simple movie recommenders server based on collaborative filtering and Keras

## Original

https://keras.io/examples/structured_data/collaborative_filtering_movielens/#introduction

## Usage:

1. Fetch this repository and Run `python ./collaborative_filtering_service` on cwd.
   - It is implemented by aiohttp so you can use its CLI to specify configs such as ip and port. Reference: https://docs.aiohttp.org/en/stable/web_quickstart.html#aiohttp-web-handler
2. Post json to `http://0.0.0.0:8080/` (default)
   - json format is like:
    ```json
    {
        "movieId" : [1340,1341,1345,1346,1347,1348,1350,1356,1358,2301,2302]
    }
    ```
   - response json format is like:
    ```json
    {"recommended_movie_ids": [318, 858, 527, 2019, 745, 1148, 50, 904, 1198, 750, 922, 260, 912, 1193, 3435, 1212, 1204, 913, 3030, 908, 1221, 1207, 2762, 1250, 923, 1178, 593, 720, 1262, 2905]}
    ```
    - movie_ids refers to `movies2.csv`. To retrieve movies information you can use this. (Both `movies2.csv` and `ratings.csv` are part of `ml-1m` on http://files.grouplens.org/datasets/movielens/)
3. `collaborative_filering_model.py` is for train. The others are just for test. You can run `request_test.py` to test the server.

## Requirements:

```
tensorflow == 2.7.0
pandas
aiohttp
requests
```
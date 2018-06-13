import numpy as np
import pandas as pd
import data_handler as dh
import paths


def calculate_item_mean(ratings: np.ndarray, item_data: np.ndarray):
    """ Calculate a mean based of data variance and offsets

    :param ratings: all ratings in the training data
    :param item_data: data related to the item whose mean we are calculating
    :return:
    """
    overall_mean = np.mean(ratings)
    overall_variance = np.var(ratings)

    item_sum = np.sum(item_data)
    item_variance = np.var(item_data)
    item_size = len(item_data)
    var_ratio = item_variance / overall_variance

    item_mean = (overall_mean * var_ratio + item_sum) / (var_ratio + item_size)
    return item_mean


def calculate_all_means(df_data: pd.DataFrame):
    """ Calculate the mean rating per movie as well as the average user offset

    :param df_data: All training data
    :return: Arrays containing the calculated values for each user and movie
    """
    user_ids = (df_data['row_id'] - 1).values
    movie_ids = (df_data['col_id'] - 1).values
    ratings = df_data['Prediction'].values

    # find the average ratings for each movie
    movie_ratings = []
    for m in range(0, paths.num_movies):
        # find entries and then ratings for movie m
        m_entries = np.where(np.equal(movie_ids, m))[0]
        m_ratings = ratings[m_entries]
        movie_mean = calculate_item_mean(ratings, m_ratings)
        movie_ratings.append(movie_mean)

    # find the offset of each rating
    rating_offsets = []
    for r in range(0, len(ratings)):
        offset = ratings[r] - movie_ratings[movie_ids[r]]
        rating_offsets.append(offset)

    rating_offsets = np.asarray(rating_offsets)
    user_offsets = []
    # calculate the mean offset for each user
    for u in range(0, paths.num_users):
        # find all entries for user u and then all their ratings
        u_entries = np.where(np.equal(user_ids, u))[0]
        u_offsets = rating_offsets[u_entries]
        user_mean = calculate_item_mean(ratings, u_offsets)
        user_offsets.append(user_mean)

    return {"mean_movie_rating" : movie_ratings, "mean_user_offsets" : user_offsets}


def predict_mean_rating(mean_predictions: dict, user: int, movie: int):
    """ Get a prediction rating based purely on the movie average rating and user offset

    :param mean_predictions: data with all the averages
    :param user: user index
    :param movie: movie index
    :return: the predicted rating
    """
    movie_ratings = mean_predictions["mean_movie_rating"]
    user_offsets = mean_predictions["mean_user_offsets"]
    rating = movie_ratings[movie] + user_offsets[user]
    rating = min(paths.max_rating, rating)
    rating = max(paths.min_rating, rating)

    return rating


# make a prediction for a single user-movie pair
def make_prediction(mean_predictions, user_features, movie_features, bu, bm, user, movie):
    rating = predict_mean_rating(mean_predictions, user, movie)
    rating += sum(user_features[user] * movie_features[:, movie]) + bu[user] + bm[movie]

    return rating


def run():
    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    train_samples: np.ndarray = dh.df_as_array(df_train_data)

    mean_predictions = calculate_all_means(df_train_data)


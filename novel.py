import numpy as np
import pandas as pd
from time import time
import logging
import logging.config
import math
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

    print("Calculating average movie ratings")
    # find the average ratings for each movie
    movie_ratings = []
    for m in range(0, paths.num_movies):
        # find entries and then ratings for movie m
        m_entries = np.where(np.equal(movie_ids, m))[0]
        m_ratings = ratings[m_entries]
        movie_mean = calculate_item_mean(ratings, m_ratings)
        movie_ratings.append(movie_mean)

    print("Calculating rating offsets")
    # find the offset of each rating
    rating_offsets = []
    for r in range(0, len(ratings)):
        offset = ratings[r] - movie_ratings[movie_ids[r]]
        rating_offsets.append(offset)

    rating_offsets = np.asarray(rating_offsets)
    user_offsets = []
    print("Calculating average user offsets")
    # calculate the mean offset for each user
    for u in range(0, paths.num_users):
        # find all entries for user u and then all their ratings
        u_entries = np.where(np.equal(user_ids, u))[0]
        u_offsets = rating_offsets[u_entries]
        user_mean = calculate_item_mean(ratings, u_offsets)
        user_offsets.append(user_mean)

    return {"mean_movie_rating" : movie_ratings, "mean_user_offsets" : user_offsets}


def predict_initial_rating(mean_predictions: dict, user: int, movie: int):
    """ Get a prediction rating for a user-movie pair
    based purely on the movie average rating and user offset

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
    rating = predict_initial_rating(mean_predictions, user, movie)
    rating += sum(user_features[user] * movie_features[:, movie]) + bu[user] + bm[movie]

    return rating


def final_predictions(mean_predictions, user_features, movie_features, bu, bm):
    prediction_matrix = np.zeros((10000, 1000))
    for u in range(0, 10000):
        for m in range(0, 1000):
            prediction = make_prediction(mean_predictions, user_features, movie_features,
                                         bu, bm, u, m)
            prediction = max(1.0, prediction)
            prediction = min(5.0, prediction)
            prediction_matrix[u][m] = prediction

    return prediction_matrix


def calculate_rmse(mean_predictions, user_features, movie_features, bu, bm, test_samples):
    """ Calculate the rmse w.r.t. every sample in the testing set """
    errors = []

    for sample in test_samples:
        rating = sample[paths.rating_id]
        predicted_rating = make_prediction(mean_predictions, user_features, movie_features,
                                           bu, bm, sample[paths.user_id], sample[paths.movie_id])
        err = rating - predicted_rating
        errors.append(err*err)

    rmse = math.sqrt(np.mean(errors))
    return rmse


def train(k, mean_predictions, user_features, movie_features, bu, bm, train_data, test_data):
    logger = logging.getLogger('sLogger')
    rmse: float = calculate_rmse(mean_predictions, user_features, movie_features, bu, bm, test_data)
    prev_rmse: float = rmse
    logger.info("Starting RMSE: {0}".format(rmse))

    for feature in range(0, k):
        print("Training feature {0}".format(feature))
        logger.info("Training feature {0}".format(feature))
        user_features[:, feature] = 0.1
        movie_features[feature] = 0.1

        tic = time()
        # train the feature
        for i in range(1, 100):
            for sample in train_data:
                user = sample[paths.user_id]
                movie = sample[paths.movie_id]

                mf = movie_features[:, movie][feature]
                uf = user_features[user][feature]

                predicted_rating = make_prediction(mean_predictions, user_features, movie_features,
                                                   bu, bm, user, movie)
                err = sample[paths.rating_id] - predicted_rating

                # update the features
                user_features[user][feature] += paths.learning_rate * (err * mf - paths.lambda_term * uf)
                movie_features[:, movie][feature] += paths.learning_rate * (err * uf - paths.lambda_term * mf)

                bu[user] += paths.learning_rate * (err - paths.lambda_term * bu[user])
                bm[movie] += paths.learning_rate * (err - paths.lambda_term * bm[movie])

            rmse = calculate_rmse(mean_predictions, user_features, movie_features, bu, bm, test_data)
            toc = time()
            iter_time = (toc - tic) / i
            logger.info('Iteration: %d, Misfit: %.8f, Improvement: %.8f, Time: %.3f'
                        % (i, rmse, prev_rmse - rmse, iter_time))
            if (rmse > prev_rmse and i > 4):
                break
            prev_rmse = rmse


def run():
    logging.config.fileConfig("logging_config.ini")
    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    train_samples: np.ndarray = dh.df_as_array(df_train_data)
    test_samples: np.ndarray = dh.df_as_array(df_test_data)

    mean_predictions = calculate_all_means(df_train_data)

    # initialize variables needed for training
    k = 80
    bu = np.zeros(paths.num_users)
    bm = np.zeros(paths.num_movies)
    user_features = np.zeros((paths.num_users, k))
    movie_features = np.zeros((k, paths.num_movies))

    train(k, mean_predictions, user_features, movie_features, bu, bm, train_samples, test_samples)

    print("Calculating predictions and writing file")
    prediction_matrix = final_predictions(mean_predictions, user_features, movie_features, bu, bm)
    dh.write_submission(prediction_matrix)


if __name__ == "__main__":
    run()

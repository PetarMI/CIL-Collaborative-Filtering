import pandas as pd
import numpy as np
from time import time
import data_handler as dh
import svd_approach as svd_base
import paths
import logging
import logging.config


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


def train(k: int, train_samples, df_test_data, U, M, bu, bi):
    """ Main function running the simple SGD approach
    """
    logger = logging.getLogger('sLogger')
    logger.info("Training for K = {0}".format(k))

    # Calculate the initial loss
    prediction_matrix = make_predictions(U, M, bu, bi)
    rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)
    logger.info("Initial loss: {0}".format(rmse))

    # initialize other variables needed for training
    alpha: float = paths.learning_rate
    lambda_term: float = paths.lambda_term

    # initialize variables used for backtracking the solution
    prev_U: np.ndarray = U
    prev_M: np.ndarray = M
    prev_bu: np.ndarray = bu
    prev_bi: np.ndarray = bi
    prev_rmse: float = rmse

    i_iter = 1
    tic = time()
    num_useless_iter = 0
    uphill_iter = 0

    print("Starting SGD algorithm")
    while(i_iter <= paths.sgd_max_iteration):
        # perform update steps
        U, M, bu, bi = sgd_update(train_samples, U, M, bu, bi, alpha, lambda_term)

        prediction_matrix = make_predictions(U, M, bu, bi)
        rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)

        # bookkeeping
        toc = time()
        iter_time = (toc - tic) / i_iter
        logger.info('Iteration: %d, Misfit: %.8f, Improvement: %.8f, Time: %.3f'
                    % (i_iter, rmse, prev_rmse - rmse, iter_time))

        # stop sgd when we see little to no improvement for 1000 iterations
        if (rmse > prev_rmse - 1e-7):
            num_useless_iter += 1
            logger.info("Useless iteration {0}".format(num_useless_iter))
        else:
            num_useless_iter = 0
        if (num_useless_iter == 10):
            break

        # revert to previous values of approximation matrices
        # if there is no improvement over the last 100 iterations
        if (rmse < prev_rmse):
            prev_U = np.copy(U)
            prev_M = np.copy(M)
            prev_bu = np.copy(bu)
            prev_bi = np.copy(bi)
            prev_rmse = rmse
            uphill_iter = 0
        else:
            uphill_iter += 1
            logger.info("Went uphill: {0}".format(uphill_iter))
            # if rmse keeps getting worse after 5 iterations revert to previous best result
            if (uphill_iter >= 5):
                logger.info("Revert iterations")
                U = np.copy(prev_U)
                M = np.copy(prev_M)
                bu = np.copy(prev_bu)
                bi = np.copy(prev_bi)

                # update learning rate so we don't miss the minimum again
                alpha /= 1.5
                uphill_iter = 0

        i_iter += 1

    # normalize best result
    prediction_matrix = make_predictions(prev_U, prev_M, prev_bu, prev_bi)
    prediction_matrix = normalize_predictions(prediction_matrix)

    rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)
    logger.info("Final RMSE for K = {0} is {1}".format(k, rmse))

    return prediction_matrix, rmse


def sgd_update(train_samples, U, M, bu, bi, l_rate, l):
    """ Perform the update step of SGD

    :param train_samples: all samples we are training w.r.t
    :param U, M: Approximation matrices
    :param l_rate: learning rate
    :param l: regularizer term
    :return: updated approximation matrices and
    """
    for i in np.random.permutation(len(train_samples)):
        user = train_samples[i][paths.user_id]
        movie = train_samples[i][paths.movie_id]
        rating = train_samples[i][paths.rating_id]

        prediction = np.dot(U[user, :], M[:, movie]) + bu[user] + bi[0, movie]
        err = rating - prediction

        # update the biases
        bu[user] += l_rate * (err - l * bu[user])
        bi[0, movie] += l_rate * (err - l * bi[0, movie])

        # update the approximation matrices
        U[user, :] += l_rate * (err * M[:, movie] - l * U[user, :])
        M[:, movie] += l_rate * (err * U[user, :] - l * M[:, movie])

    return U, M, bu, bi


def make_predictions(U: np.ndarray, M: np.ndarray, bu: np.ndarray, bi: np.ndarray) -> np.ndarray:
    """ Make the prediction based on the approximation matrices

    :param U, M: Approximation matrices
    :param bu: User biases
    :param bi: Movie biases
    :return predictions: The approximation matrix we get after dot product of truncated U and V
    """
    prediction_matrix: np.ndarray = np.dot(U, M)
    # add the biases
    prediction_matrix += bu + bi

    return prediction_matrix


def normalize_predictions(prediction_matrix: np.ndarray) -> np.ndarray:
    """ Make sure final predictions are between 1 and 5

    :param prediction_matrix: The SGD matrix
    :return: normalized matrix
    """
    prediction_matrix[prediction_matrix > paths.max_rating] = paths.max_rating
    prediction_matrix[prediction_matrix < paths.min_rating] = paths.min_rating

    return prediction_matrix


def init_baseline(df_data, mean_predictions, k):
    prediction_matrix = np.zeros((10000, 1000))
    for u in range(0, 10000):
        for m in range(0, 1000):
            prediction = predict_initial_rating(mean_predictions, u, m)
            prediction = max(1.0, prediction)
            prediction = min(5.0, prediction)
            prediction_matrix[u][m] = prediction

    user_ids: pd.Series = df_data["row_id"] - 1
    movie_ids: pd.Series = df_data["col_id"] - 1

    prediction_matrix[user_ids, movie_ids] = df_data['Prediction']

    u, vh = svd_base.perform_svd(prediction_matrix)

    u_prime = u[:, :k]
    vh_prime = vh[:k, :]

    return u_prime, vh_prime


def cross_validation(df_train_data, train_samples, df_test_data, mean_predictions):
    ks = [5, 7, 9, 10, 11, 12, 15, 20]
    rmses = []

    print("Starting cross-validation")
    for k in ks:
        print("Cross-validating for K = {0}".format(k))
        rmse = execute_approach(k, df_train_data, train_samples, df_test_data, mean_predictions)
        rmses.append(rmse)

    print(rmses)


def execute_approach(k, df_train_data, train_samples, df_test_data, mean_predictions):
    print("Initialize parameters")

    U, M = init_baseline(df_train_data, mean_predictions, k)
    bu: np.ndarray = np.zeros([paths.num_users, 1])
    bi: np.ndarray = np.zeros([1, paths.num_movies])

    prediction_matrix, rmse = train(k, train_samples, df_test_data, U, M, bu, bi)

    # save data
    assert (prediction_matrix.shape == (paths.num_users, paths.num_movies))
    dh.write_submission(prediction_matrix)

    return rmse


def run():
    logging.config.fileConfig("logging_config.ini")

    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    print("Calculating initialization data")
    mean_predictions = calculate_all_means(df_train_data)
    train_samples: np.ndarray = dh.df_as_array(df_train_data)

    # Perform either cross validation or a single run using best result
    # cross_validation(df_train_data, train_samples, df_test_data, mean_predictions)
    k = 10
    execute_approach(k, df_train_data, train_samples, df_test_data, mean_predictions)


if __name__ == "__main__":
    run()
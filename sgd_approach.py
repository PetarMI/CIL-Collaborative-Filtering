import pandas as pd
import numpy as np
from time import time
import data_handler as dh
import svd_approach as svd_base
import paths
import logging, logging.config


def train(df_train_data: pd.DataFrame, df_test_data: pd.DataFrame):
    """ Main function running the simple SGD approach
    """
    logger = logging.getLogger('sLogger')

    print("Initializing state of approximation matrices")
    # Initialize the starting matrices using SVD
    k = 5
    U, M = init_svd_baseline(df_train_data, k)
    bu: np.ndarray = np.zeros([paths.num_users, 1])
    bi: np.ndarray = np.zeros([1, paths.num_movies])
    mu: float = df_train_data['Prediction'].mean()

    # Calculate the initial loss
    prediction_matrix = make_predictions(U, M, mu, bu, bi)
    rmse = svd_base.calc_rmse(df_train_data, prediction_matrix)
    print("Initial loss: {0}".format(rmse))

    # initialize other variables needed for training
    train_samples: np.ndarray = dh.df_as_array(df_train_data)
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
        U, M, bu, bi = sgd_update(train_samples, U, M, mu, bu, bi, alpha, lambda_term)

        prediction_matrix = make_predictions(U, M, mu, bu, bi)
        rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)

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

        toc = time()
        iter_time = (toc - tic) / i_iter
        logger.info('Iteration: %d, Misfit: %.8f, Time: %.3f' % (i_iter, rmse, iter_time))
        # print('Iteration: %d, Misfit: %.6f' % (i_iter, rmse))
        # print('Average time per iteration: %.4f' % ((toc - tic) / i_iter))
        i_iter += 1

    prediction_matrix = make_predictions(prev_U, prev_M, mu, prev_bu, prev_bi)    
    # normalize best result
    prediction_matrix[prediction_matrix > paths.max_rating] = paths.max_rating
    prediction_matrix[prediction_matrix < paths.min_rating] = paths.min_rating

    assert (prediction_matrix.shape == (paths.num_users, paths.num_movies))
    dh.write_submission(prediction_matrix)


def sgd_update(train_samples, U, M, mu, bu, bi, l_rate, l):
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

        prediction = np.dot(U[user, :], M[:, movie]) + mu + bu[user] + bi[0, movie]
        err = rating - prediction

        # update the biases
        bu[user] += l_rate * (err - l * bu[user])
        bi[0, movie] += l_rate * (err - l * bi[0, movie])

        # update the approximation matrices
        U[user, :] += l_rate * (err * M[:, movie] - l * U[user, :])
        M[:, movie] += l_rate * (err * U[user, :] - l * M[:, movie])

    return U, M, bu, bi


def make_predictions(U: np.ndarray, M: np.ndarray, mu: float, bu: np.ndarray, bi: np.ndarray) -> np.ndarray:
    """ Make the prediction based on the approximation matrices

    :param U, M: Approximation matrices
    :param mu: Mean rating for all movies
    :param bu: User biases
    :param bi: Movie biases
    :return predictions: The approximation matrix we get after dot product of truncated U and V
    """
    prediction_matrix: np.ndarray = np.dot(U, M)
    # add the biases
    prediction_matrix += mu + bu + bi

    return prediction_matrix


def init_random_baseline(k: int):
    """ Baseline approximation matrices with random values
        drawn from a Normal distribution

    :param k: Number of latent features
    :return: The random approximation matrices
    """
    U = np.random.normal(scale=1. / k, size=(paths.num_users, k))
    M = np.random.normal(scale=1. / k, size=(k, paths.num_movies))

    return U, M


def init_svd_baseline(df_data: pd.DataFrame, k: int):
    """ Prepare the baseline for first iteration of SGD
    Baseline is the approximation matrices inferred from the SVD approach

    :param df_data: Training data that we initialize from
    :param k: Number of features
    :return: The approximation matrices
    """
    A: np.ndarray = svd_base.fill_averages(df_data)
    mu: float = df_data['Prediction'].mean()
    u, vh = svd_base.perform_svd(A-mu)

    u_prime = u[:, :k]
    vh_prime = vh[:k, :]

    return u_prime, vh_prime

def run():
    logging.config.fileConfig("logging_config.ini")

    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    train(df_train_data, df_test_data)


if __name__ == "__main__":
    run()

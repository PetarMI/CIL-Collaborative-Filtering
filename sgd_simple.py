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

    dh.log(logger, "Initializing state of approximation matrices", False)
    # Initialize the starting matrices using SVD
    k = 5
    U, M = init_svd_baseline(df_train_data, k)

    # Calculate the initial loss
    prediction_matrix: np.ndarray = np.dot(U, M)
    rmse = svd_base.calc_rmse(df_train_data, prediction_matrix)
    dh.log(logger, "Initial loss: {0}".format(rmse), False)

    # initialize other variables needed for training
    train_samples: np.ndarray = dh.df_as_array(df_train_data)
    alpha: float = paths.learning_rate
    lambda_term: float = paths.lambda_term

    # initialize variables used for backtracking the solution
    prev_U: np.ndarray = np.copy(U)
    prev_M: np.ndarray = np.copy(M)
    prev_rmse: float = rmse

    i_iter = 1
    tic = time()
    num_useless_iter = 0

    dh.log(logger, "Starting SGD algorithm", False)
    while(i_iter <= paths.sgd_max_iteration):
        # perform update steps
        U, M = sgd_update(train_samples, U, M, alpha, lambda_term)

        prediction_matrix = np.dot(U, M)
        rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)

        # stop sgd when we see little to no improvement for 1000 iterations
        if (rmse > prev_rmse - 1e-7):
            num_useless_iter += 1
            dh.log(logger, "Useless iteration - {0}".format(num_useless_iter), True)
        else:
            num_useless_iter = 0
        if (num_useless_iter == 10):
            break

        # revert to previous values of approximation matrices
        # if there is no improvement over the last 100 iterations
        if (rmse < prev_rmse):
            prev_U = np.copy(U)
            prev_M = np.copy(M)
            prev_rmse = rmse
        else:
            U = np.copy(prev_U)
            M = np.copy(prev_M)

            dh.log(logger, "Revert iterations", True)
            # update learning rate so we don't miss the minimum
            alpha /= 1.5

        toc = time()
        dh.log(logger, 'Iteration: %d, Misfit: %.6f' % (i_iter, rmse), False)
        dh.log(logger, 'Average time per iteration: %.4f' % ((toc - tic) / i_iter), True)
        i_iter += 1

    # normalize best result
    prediction_matrix[prediction_matrix > paths.max_rating] = paths.max_rating
    prediction_matrix[prediction_matrix < paths.min_rating] = paths.min_rating

    assert (prediction_matrix.shape == (paths.num_users, paths.num_movies))
    dh.write_submission(prediction_matrix)


def sgd_update(train_samples, U, M, l_rate, l):
    """ Perform the update step of SGD

    :param train_samples: all samples we are training w.r.t
    :param U: Approximation matrix
    :param M: Approximation matrix
    :param l_rate: learning rate
    :param l: regularizer term
    :return: updated approximation matrices and
    """
    for i in np.random.permutation(len(train_samples)):
        user = train_samples[i][paths.user_id]
        movie = train_samples[i][paths.movie_id]
        rating = train_samples[i][paths.rating_id]

        prediction = np.dot(U[user, :], M[:, movie])
        err = rating - prediction

        # update the approximation matrices
        U[user, :] += l_rate * (err * M[:, movie] - l * U[user, :])
        M[:, movie] += l_rate * (err * U[user, :] - l * M[:, movie])

    return U, M


# TODO annotate return type
def init_svd_baseline(df_data: pd.DataFrame, k: int):
    """ Prepare the baseline for first iteration of SGD
        Baseline is the approximation matrices inferred from the SVD approach

    :param df_data: Training data that we initialize from
    :param k: Number of features
    :return: The approximation matrices
    """
    A: np.ndarray = svd_base.fill_averages(df_data)
    u, vh = svd_base.perform_svd(A)

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

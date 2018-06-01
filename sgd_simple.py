import pandas as pd
import numpy as np
from time import time
import data_handler as dh
import svd_approach as svd_base
import paths


def train(df_train_data: pd.DataFrame, df_test_data: pd.DataFrame):
    """ Main function running the simple SGD approach

    :return:
    """
    print("Initializing state of approximation matrices")
    # Initialize the starting matrices using SVD
    k = 15
    u, vh = init_baseline(df_train_data)
    U, M = set_features(k, u, vh)

    # Calculate the initial loss
    prediction_matrix = make_predictions(U, M)
    rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)
    print("Initial loss: {0}".format(rmse))

    # initialize other variables needed for training
    train_samples: np.ndarray = dh.df_as_array(df_train_data)
    alpha: float = paths.learning_rate
    lambda_term: float = paths.lambda_term
    i_iter = 1
    tic = time()

    print("Starting SGD algorithm")
    while(i_iter <= paths.sgd_max_iteration):
        # perform update steps
        U, M = sgd_update(train_samples, U, M, alpha, lambda_term)

        # check results
        prediction_matrix = make_predictions(U, M)
        rmse = svd_base.calc_rmse(df_test_data, prediction_matrix)

        i_iter += 1
        if i_iter % 100 == 0:
            toc = time()
            print('Iteration: %d, Misfit: %.5f' % (i_iter, rmse))
            print('Average time per iteration: %.4f' % ((toc - tic) / i_iter))

    # normalize best result
    prediction_matrix[prediction_matrix > paths.max_rating] = paths.max_rating
    prediction_matrix[prediction_matrix < paths.min_rating] = paths.min_rating

    assert (prediction_matrix.shape == (paths.num_users, paths.num_movies))
    dh.write_submission(prediction_matrix)


def sgd_update(train_samples, U, M, alpha, l):
    """ Perform the update step of SGD

    :param train_samples: all samples we are training w.r.t
    :param U: Approximation matrix
    :param M: Approximation matrix
    :param alpha: learning rate
    :param l: regularizer term
    :return: updated approximation matrices
    """
    for i in np.random.permutation(len(train_samples)):
        user = train_samples[i][paths.user_id]
        movie = train_samples[i][paths.movie_id]
        rating = train_samples[i][paths.rating_id]

        prediction = np.dot(U[user, :], M[:, movie])
        err = rating - prediction

        # update step
        U[user, :] += alpha * (err * M[:, movie] - l * U[user, :])
        M[:, movie] += alpha * (err * U[user, :] - l * M[:, movie])

        return U, M


# TODO is it gonna be quicker if we multiply just training data in a for loop
def make_predictions(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """ Make the prediction based on the approximation matrices

    :param P: Approximation matrix
    :param Q: Approximation matrix
    :return predictions: The approximation matrix we get after dot product of truncated U and V
    """
    prediction_matrix: np.ndarray = np.dot(P, Q)
    return prediction_matrix


# TODO annotate return type
def init_baseline(df_data: pd.DataFrame):
    """ Prepare the baseline for first iteration of SGD
        Baseline is the approximation matrices inferred from the SVD approach

    :return:
    """
    A: np.ndarray = svd_base.fill_averages(df_data)
    U, Vh = svd_base.perform_svd(A)

    return U, Vh


# TODO annotate return type
def set_features(k: int, u: np.ndarray, vh: np.ndarray):
    """ Choose how many features to use

    :param k: Number of features
    :param u: Approximation matrix
    :param vh: Approximation matrix
    :return: The two matrices
    """
    u_prime = u[:, :k]
    vh_prime = vh[:k, :]

    return u_prime, vh_prime


def run():
    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    train(df_train_data, df_test_data)


if __name__ == "__main__":
    run()

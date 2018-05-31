import pandas as pd
import numpy as np
import data_handler as dh
import svd_approach as svd_base
import paths


# TODO is it gonna be quicker if we multiply just training data in a for loop
def make_predictions(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """ Make the prediction based on the approximation matrices

    :param P: Approximation matrix
    :param Q: Approximation matrix
    :return predictions: The approximation matrix we get after dot product of truncated U and V
    """
    prediction_matrix: np.ndarray = np.dot(P, Q)
    return prediction_matrix


def train():
    """ Main function running the simple SGD approach

    :return:
    """
    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    print("Initializing state of approximation matrices")
    k = 10
    u, vh = init_baseline(df_train_data)
    P, Q = set_features(k, u, vh)
    assert(P.shape == (paths.num_users, k))
    assert(Q.shape == (k, paths.num_movies))

    print("Starting SGD algorithm")
    initial_prediction = make_predictions(P, Q)
    rmse = svd_base.calc_rmse(df_test_data, initial_prediction)
    print("Initial loss: {0}".format(rmse))


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

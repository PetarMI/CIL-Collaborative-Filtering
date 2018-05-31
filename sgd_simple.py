import pandas as pd
import numpy as np
import data_handler as dh
import svd_approach as svd_base
import paths


def train():
    """ Main function running the simple SGD approach

    :return:
    """
    print("Initializing baseline solution")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    # get the starting P and Q matrices
    k = 10
    u, vh = init_baseline(df_train_data)
    P, Q = set_features(k, u, vh)
    assert(P. shape == (paths.num_users, k))
    assert(Q.shape == (k, paths.num_movies))


    print("Starting SGD algorithm")


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
    u_prime = u[:, 0:k + 1]
    vh_prime = vh[0:k + 1, :]

    return u_prime, vh_prime

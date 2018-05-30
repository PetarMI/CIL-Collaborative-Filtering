import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import data_handler as dh
import paths
import matplotlib.pyplot as plt


def fill_averages(df_data: pd.DataFrame) -> np.ndarray:
    """ Construct a rating matrix with average ratings inserted in positions
        with not rating

    :param df_data: The processed data that we read from the csv file
    :return avg_matrix: matrix with movie averages filled in
    """
    movie_averages = df_data.groupby('col_id')['Prediction'].mean()

    user_ids: pd.Series = df_data["row_id"] - 1
    movie_ids: pd.Series = df_data["col_id"] - 1

    # init a matrix of all zeros and fill in ratings
    avg_matrix: np.ndarray = np.zeros((paths.num_users, paths.num_movies))
    avg_matrix += movie_averages.as_matrix()
    avg_matrix[user_ids, movie_ids] = df_data['Prediction']

    return avg_matrix


# TODO annotate return type
def perform_svd(A: np.ndarray):
    """ Perform SVD on a given matrix and process the two approximation matrices

    :param A: Matrix to perform SVD on
    :return u_prime, vh_prime: Tuple containing the approximation matrices
    """
    u, s, vh = np.linalg.svd(A, full_matrices=False)

    s_diag = np.diag(np.sqrt(s))
    u_prime = np.dot(u, s_diag)
    vh_prime = np.dot(s_diag, vh)

    return u_prime, vh_prime


def make_predictions(k: int, U: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    """ Make an approximation from SVD using first K features

    :param k: Number of features to use
    :param U:
    :param Vh:
    :return predictions: The approximation matrix we get after dot product of truncated U and V
    """
    # truncate approximation matrices using the first K features
    u_prime = U[:, 0:k + 1]
    vh_prime = Vh[0:k + 1, :]

    prediction_matrix = np.dot(u_prime, vh_prime)
    return prediction_matrix


def calc_rmse(df_data: pd.DataFrame, prediction_matrix: np.ndarray) -> float:
    """ Works by constructing both matrices and then taking only relevant entries

    :param df_data: Data we calculate RMSE w.r.t
    :param prediction_matrix: The prediction matrix we have
    :return:
    """
    user_ids: pd.Series = df_data["row_id"] - 1
    movie_ids: pd.Series = df_data["col_id"] - 1

    test_data_matrix: coo_matrix = dh.data_as_matrix(df_data).todense()

    err: np.ndarray = (test_data_matrix - prediction_matrix)[user_ids, movie_ids]
    loss: float = np.sqrt(np.sum(np.asarray(err) ** 2) / len(user_ids))

    return loss


def cross_validation():
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    A: np.ndarray = fill_averages(df_train_data)

    U, Vh = perform_svd(A)

    min_k = 2
    max_k = 50

    print("Starting cross validation")

    ks = []
    errs = []

    # Winning K = 10
    for k in range(min_k, max_k + 1):
        prediction_matrix = make_predictions(k, U, Vh)
        err = calc_rmse(df_test_data, prediction_matrix)
        print("K = {0}, RMSE = {1}".format(k, err))
        ks.append(k)
        errs.append(err)

    plt.plot(ks, errs)
    plt.show()


def execute_approach():
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)

    A: np.ndarray = fill_averages(df_data)
    U, Vh = perform_svd(A)

    # K = 10 was the winning value from the cross validation
    k = 10

    prediction_matrix = make_predictions(k, U, Vh)
    assert(prediction_matrix.shape == (10000, 1000))
    dh.write_submission(prediction_matrix)


def run():
    # cross_validation()
    execute_approach()


if __name__ == "__main__":
    run()

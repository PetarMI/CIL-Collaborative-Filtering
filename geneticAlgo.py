import pandas as pd
import numpy as np
import random
from time import time
import data_handler as dh
import svd_approach as svd_base
import paths
import logging, logging.config


def train(k: int, df_train_data: pd.DataFrame, df_test_data: pd.DataFrame):
    """ Main function running genetic algorithm
    """
    logger = logging.getLogger('sLogger')
    logger.info("Training for K = {0}".format(k))

    print("Initializing first generation")
    # Initialize the first generation at random
    U, M, bu, bi = init_random_baseline(k)
    mu: float = df_train_data['Prediction'].mean()
    prev_rmse: float = 0
    overallBest: float = 4.0
    bestU: np.ndarray = np.zeros([paths.num_users, k])
    bestM: np.ndarray = np.zeros([k, paths.num_movies])
    bestbu: np.ndarray = np.zeros([paths.num_users, 1])
    bestbi: np.ndarray = np.zeros([1, paths.num_movies])

    tic = time()

    for i in range(0,paths.num_generations):
        new_U = np.copy(U)
        new_M = np.copy(M)
        new_bu = np.copy(bu)
        new_bi = np.copy(bi)
        ratings, best_rmse, bestInd = rate(df_train_data, U, M, mu, bu, bi)

        if overallBest > best_rmse:
            bestU = np.copy(U[bestInd])
            bestM = np.copy(M[bestInd])
            bestbu = np.copy(bu[bestInd])
            bestbi = np.copy(bi[bestInd])
            overallBest = best_rmse

        for j in range(0,paths.gen_size):
            p1 = selection(ratings)
            p2 = selection(ratings)
            new_U[j], new_M[j], new_bu[j], new_bi[j] = crossover(U[p1], M[p1], bu[p1], bi[p1], \
                U[p2], M[p2], bu[p2], bi[p2], k)

        U = np.copy(new_U)
        M = np.copy(new_M)
        bu = np.copy(new_bu)
        bi = np.copy(new_bi)

        toc = time()
        iter_time = (toc - tic) / (i+1)
        logger.info('Iteration: %d, Misfit: %.8f, Improvement: %.8f, Time: %.3f'
            % (i+1, best_rmse, prev_rmse - best_rmse, iter_time))

        prev_rmse = best_rmse

    # normalize best result
    prediction_matrix = make_predictions(bestU, bestM, mu, bestbu, bestbi)
    prediction_matrix = normalize_predictions(prediction_matrix)

    rmse = svd_base.calc_rmse(df_train_data, prediction_matrix)
    logger.info("Final RMSE for K = {0} is {1}".format(k, rmse))

    # save data
    assert (prediction_matrix.shape == (paths.num_users, paths.num_movies))
    dh.write_submission(prediction_matrix)


def mutation(U, M, bu, bi, k):
    if (random.uniform(0,1) <= paths.mutation_rate):
        x = random.randint(0, U.shape[0]-1)
        y = random.randint(0, U.shape[1]-1)
        U[x,y] = np.random.normal(scale=1. / k)

    if (random.uniform(0,1) <= paths.mutation_rate):
        x = random.randint(0, M.shape[0]-1)
        y = random.randint(0, M.shape[1]-1)
        M[x,y] = np.random.normal(scale=1. / k)

    if (random.uniform(0,1) <= paths.mutation_rate):
        x = random.randint(0, bu.shape[0]-1)
        y = random.randint(0, bu.shape[1]-1)
        bu[x,y] = np.random.normal(scale=1. / k)

    if (random.uniform(0,1) <= paths.mutation_rate):
        x = random.randint(0, bi.shape[0]-1)
        y = random.randint(0, bi.shape[1]-1)
        bi[x,y] = np.random.normal(scale=1. / k)

    return U, M, bu, bi

def crossover2(U1, M1, bu1, bi1, U2, M2, bu2, bi2, k):
    half = k // 2
    U = np.concatenate((U1[:,0:half+1], U2[:, half+1:]), axis = 1)
    M = np.concatenate((M1[0:half+1,:], M2[half+1:,:]), axis = 0)
    bu = (bu1+bu2)/2
    bi = (bi1+bi2)/2
    
    # Mutate
    U, M, bu, bi = mutation (U, M, bu, bi, k)

    return U, M, bu, bi

def crossover(U1, M1, bu1, bi1, U2, M2, bu2, bi2, k):
    U = (U1+U2)/2
    M = (M1+M2)/2
    bu = (bu1+bu2)/2
    bi = (bi1+bi2)/2
    
    # Mutate
    U, M, bu, bi = mutation (U, M, bu, bi, k)

    return U, M, bu, bi


def selection(ratings):
    n = len(ratings)
    sum = n*(n+1) / 2

    choice = random.randint(1, sum)

    tuples = [(ratings[i], i) for i in range(0, len(ratings))]
    tuples.sort()

    running_sum = 0
    index = len(ratings)
    for curr in tuples:
        running_sum = running_sum + index
        if choice <= running_sum:
            return curr[1]
        index = index - 1


def rate(df_train_data: pd.DataFrame, U: np.ndarray, M: np.ndarray, mu: float, bu: np.ndarray, bi: np.ndarray):

    rmses: np.ndarray = np.zeros(paths.gen_size)
    best: float = 0.0
    ind = 1

    for i in range(0,paths.gen_size):
        prediction_matrix = make_predictions(U[i], M[i], mu, bu[i], bi[i])
        rmses[i] = svd_base.calc_rmse(df_train_data, prediction_matrix)
        if ((i == 1) or (best > rmses[i])):
            ind = i
            best = rmses[i]

    return rmses, best, ind


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


def normalize_predictions(prediction_matrix: np.ndarray) -> np.ndarray:
    """ Make sure final predictions are between 1 and 5

    :param prediction_matrix: The SGD matrix
    :return: normalized matrix
    """
    prediction_matrix[prediction_matrix > paths.max_rating] = paths.max_rating
    prediction_matrix[prediction_matrix < paths.min_rating] = paths.min_rating

    return prediction_matrix


def init_random_baseline(k: int):
    """ Baseline approximation matrices with random values
        drawn from a Normal distribution

    :param k: Number of latent features
    :return: The random approximation matrices
    """
    U = np.random.normal(scale=1. / k, size=(paths.gen_size, paths.num_users, k))
    M = np.random.normal(scale=1. / k, size=(paths.gen_size, k, paths.num_movies))
    bu: np.ndarray = np.random.normal(scale=1., size=(paths.gen_size, paths.num_users, 1))
    bi: np.ndarray = np.random.normal(scale=1., size=(paths.gen_size, 1, paths.num_movies))

    return U, M, bu, bi


def cross_validation(df_train_data: pd.DataFrame, df_test_data: pd.DataFrame):
    ks = [4, 5, 6, 7, 8, 9, 10, 11, 15]

    for k in ks:
        train(k, df_train_data, df_test_data)


def run():
    logging.config.fileConfig("logging_config.ini")

    print("Processing data")
    df_data: pd.DataFrame = dh.read_data(paths.total_dataset_location)
    data_dict: dict = dh.split_original_data(df_data, 0.1)

    df_train_data: pd.DataFrame = data_dict["train_data"]
    df_test_data: pd.DataFrame = data_dict["test_data"]

    # cross_validation(df_train_data, df_test_data)
    # assign the best result from cross validation to K
    K = 10
    train(K, df_train_data, df_test_data)


if __name__ == "__main__":
    run()

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, bmat
from numpy.random import randint, choice
from numpy.linalg import svd
from time import time
from os.path import isfile
import matplotlib.pyplot as plt


class collabrative_filtering:

    def __init__(self, traning_data, user_N=10000, item_N=1000, **kwargs):

        self.k = kwargs.get('k', 10)

        self.user_N = user_N
        self.item_N = item_N

        self.train_data = traning_data
        self.user_ids = self.train_data['row_id'] - 1
        self.item_ids = self.train_data['col_id'] - 1

        self.r = coo_matrix((self.train_data, (self.user_ids, self.item_ids)))

        self.baseline = self.creat_baseline()

    def creat_baseline(self):
        '''
        Prepare the baseline solution
        :return:
        A: baseline matrix with averages filled in where ratings are missing
        '''

        self.mean_per_item = self.train_data.groupby('col_id')['Prediction'].mean().as_matrix()

        A = coo_matrix((self.train_data['Prediction'], (self.user_ids, self.item_ids))).todense()
        A = A + self.mean_per_item
        A[self.user_ids, self.item_ids] = self.train_data['Prediction']

        return A

    def svd_decomp(self):
        # Perform SVD on baseline matrix
        u, s, vh = np.linalg.svd(self.baseline, full_matrices=False)

        s_diag = np.diag(np.sqrt(s))
        u_prime = np.dot(u, s_diag)
        vh_prime = np.dot(vh, s_diag)

        return u_prime[:, 0:self.k], vh_prime[0:self.k, :]

    def train(self, alpha, beta, max_iter, epsilon):

        # Initialize p & q with SVD
        p, q = self.svd_decomp()

        # Initialize bias and mu
        bu = np.zeros([self.user_N, 1])
        bi = np.zeros([1, self.item_N])
        mu = np.mean(self.mean_per_item)

        prediction = np.dot(q, p) + mu + bu + bi

        loss, err = self.rms(prediction)

        print('Initial Loss: %.5f' % loss)
        print('*' * 60)
        i_iter = 1
        tic = time()
        trace = []

        while i_iter <= max_iter and loss >= epsilon:
            loss_old = loss.copy()
            self.sgd()
            loss, err = self.rms(prediction)
            trace.append([i_iter, loss])
            if i_iter % 100 == 0:
                toc = time()
                print('Iteration: %d, Misfit: %.5f, Sample: %d' % (i_iter, loss, sample))
                print('Average time per iteration: %.4f' % ((toc - tic) / i_iter))
                print('*' * 60)

            if np.abs(loss_old - loss) <= 1e-6:
                break
            else:
                i_iter += 1

        return p, q, np.asarray(trace)

    def sgd(self):


    def rms(self, prediction):

        # Calculate the total loss with respect to the whole training data
        err = (self.r.todense() - prediction)[self.user_ids, self.item_ids]

        loss = np.sqrt(np.sum(np.asarray(err) ** 2) / len(self.user_ids))

        err_matrix = coo_matrix((np.asarray(err)[0], (self.user_ids, self.item_ids))).todense()

        return loss, err_matrix




if __name__ == '__main__':

    training_dataset = pd.read_csv('./data_train_post.csv')
    cf = collabrative_filtering(training_dataset)
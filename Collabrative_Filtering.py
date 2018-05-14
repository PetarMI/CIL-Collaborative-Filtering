import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from numpy.random import randint
from numpy.linalg import svd
from time import time



class collabrative_filtering:

    def __init__(self, traning_data, user_N=10000, item_N=1000, **kwargs):

        self.k = kwargs.get('k', 10)

        self.user_N = user_N
        self.item_N = item_N

        self.train_data = traning_data
        self.user_ids = self.train_data['row_id'] - 1
        self.item_ids = self.train_data['col_id'] - 1

        self.r = coo_matrix((self.train_data['Prediction'], (self.user_ids, self.item_ids)))

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

    def _svd_decomp(self):
        # Perform SVD on baseline matrix
        u, s, vh = svd(self.baseline, full_matrices=False)

        s_diag = np.diag(np.sqrt(s))
        u_prime = np.dot(u, s_diag)
        vh_prime = np.dot(s_diag, vh)

        return u_prime[:, 0:self.k], vh_prime[0:self.k, :]
    
    def findK(self):
        traces = []
        tic = time()
        
        for currentk in range(1, 1000):
            u,v = self._svd_decomp(currentk)
            
            prediction = np.dot(u,v) 
            
            err = (self.r.todense() - prediction)[self.user_ids, self.item_ids]
            loss = np.sqrt(np.sum(np.asarray(err) ** 2) / len(self.user_ids))
        
            traces.append([currentk, loss])
            
            if currentk % 50 == 0:
                print("Number of iterations passed %d" % (currentk))
                toc = time()
                print("Average time per iteration is %f" % ((toc - tic) / currentk) )
        return np.asarray(traces)

    def train(self, alpha, beta, max_iter, epsilon, sample):

        self.alpha = alpha
        self.beta = beta
        self.sample = sample

        # Initialize p & q with SVD
        self.q, self.p = self._svd_decomp()

        # Initialize bias and mu
        self.bu = np.zeros([self.user_N, 1])
        self.bi = np.zeros([1, self.item_N])
        self.mu = np.mean(self.mean_per_item)

        self.prediction = self.get_prediction()

        self.loss, self.err = self.rms(self.prediction)

        print('Initial Loss: %.5f' % self.loss)
        print('*' * 60)
        i_iter = 1
        tic = time()
        trace = []

        while i_iter <= max_iter and self.loss >= epsilon:
            loss_old = self.loss.copy()
            self._sgd()
            self.prediction = self.get_prediction()
            self.loss, self.err = self.rms(self.prediction)
            trace.append([i_iter, self.loss])
            if i_iter % 100 == 0:
                toc = time()
                print('Iteration: %d, Misfit: %.5f, Sample: %d' % (i_iter, self.loss, sample))
                print('Average time per iteration: %.4f' % ((toc - tic) / i_iter))
                print('*' * 60)

            if np.abs(loss_old - self.loss) <= 1e-6:
                break
            else:
                i_iter += 1

        return self.prediction, np.asarray(trace)

    def _sgd(self):
        rand_ids = randint(0, len(self.user_ids), self.sample)
        for rand_id in rand_ids:
            user, item = (self.user_ids[rand_id], self.item_ids[rand_id])

            self.bu[user] += self.alpha * (self.err[user, item] - self.beta * self.bu[user])
            self.bi[0, item] += self.alpha * (self.err[user, item] - self.beta * self.bi[0, item])

            self.p[:, item] += self.alpha * (self.err[user, item] * self.q[user, :].T - self.beta * self.p[:, item])
            self.q[user, :] += self.alpha * (self.err[user, item] * self.p[:, item].T - self.beta * self.q[user, :])

    def get_prediction(self):
        return np.dot(self.q, self.p) + self.mu + self.bu + self.bi
    
    def rms(self, prediction):

        # Calculate the total loss with respect to the whole training data
        err = (self.r.todense() - prediction)[self.user_ids, self.item_ids]

        loss = np.sqrt(np.sum(np.asarray(err) ** 2) / len(self.user_ids))

        err_matrix = coo_matrix((np.asarray(err)[0], (self.user_ids, self.item_ids))).todense()

        return loss, err_matrix


if __name__ == '__main__':

    training_dataset = pd.read_csv('./data/data_train_post.csv')
    cf = collabrative_filtering(training_dataset, k=50)

    alpha = 0.02
    beta = 0.002
    epsilon = 1e-3
    max_iter = 5000
    sample = 100

    cf.train(alpha, beta, max_iter, epsilon, sample)
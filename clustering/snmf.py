import numpy as np
import numpy.linalg as LA
import scipy.io as sio # not working for me
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
#from scipy.stats import entropy
from time import time
import random, math

# magic numbers
_smallnumber = 1E-6

class SNMF():

    """
    Input:
      -- V: m x n matrix, the dataset

    Optional Input/Output:
      -- l: penalty lambda (trade-off parameter between the regularization term and the loss term.)
    
      -- w_init: basis matrix with size m x r
      -- h_init: weight matrix with size r x n  (r is the number of cluster)
      -- Output: w, h
    """
    def __init__(self, x, h_init = None, r = 2, batch_number = 10, max_iter = 100):
        self.x = x.todense()
        self.r = r
        self.max_iter = max_iter
        
        print("Constructor call: The matrix's row and column are: ", self.x.shape[0], self.x.shape[1], "total iteration: ", self.max_iter)
        
        self.batch_number = batch_number
        self.batch_number_range = self.x.shape[0]
        self.mini_batch_size = math.ceil(x.shape[0] / self.batch_number)
        self.batch_x = np.asmatrix(np.zeros((self.mini_batch_size,self.mini_batch_size)))
        print("Constructor call: Batch number is : ", batch_number, " with mini_batch_size: ", self.mini_batch_size, "batch_x has shape:", self.batch_x.shape)

        self.h = h_init
        self.errors = np.zeros(self.max_iter)

    def frobenius_norm(self):
        """ Euclidean error between x and h * h.T """

        if hasattr(self, 'h'):  # if it has attributes w and h
            error = LA.norm(self.x - self.h*self.h.T)
        else:
            error = None
        return error

    def bgd_solver(self, alpha = 0.001, eps = None, debug = None):
        if(self.batch_number == 1):        # normal MUR
            for iter in range(self.max_iter):
                self.errors[iter] = LA.norm(self.x - self.h * self.h.T)
                numerator = self.x*self.h
                denominator = (((self.h*self.h.T)*self.h) + 2 ** -8)
                self.h = np.multiply(self.h, np.divide(numerator, denominator))

        else:
            batch_h = np.asmatrix(np.zeros((self.mini_batch_size,self.r)))
            for iter in range(self.max_iter):  # stochastic MUR     
                self.errors[iter] = np.linalg.norm(self.x - self.h * self.h.T, 'fro') # record error
                tmp_list = self.generate_random_numbers(upper_bound = self.batch_number_range, num = self.mini_batch_size)                
                
                # an ugly matrix to create batch matrix
                i = 0
                while i < len(tmp_list):
                    j = i
                    batch_h[i,:] = self.h[tmp_list[i],:]
                    while j < len(tmp_list):
                        self.batch_x[i,j] = self.x[tmp_list[i],tmp_list[j]]
                        self.batch_x[j,i] = self.x[tmp_list[i],tmp_list[j]]
                        j += 1
                    i += 1

                grad = 4 * (batch_h * batch_h.T * batch_h - self.batch_x * batch_h)
                # print("grad", grad)
                update = batch_h - alpha * grad

                i = 0
                while i < len(tmp_list):
                    j = 0
                    count = 0
                    while j < update.shape[1]:
                        if update[i,j] < 0:
                            update[i,j] = 0
                            count += 1
                        j += 1

                    self.h[tmp_list[i],:] = update[i,:]   
                    i += 1
        return self.h

    def get_error_trend(self):
        return self.errors

    # generate a list of random number from range [0, range], with size num
    def generate_random_numbers(self, upper_bound, num):
        seq = list(range(0,upper_bound))
        return random.sample(seq,num)


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
    def __init__(self, x, h_init = None, r = 2, max_iter = 100):
        self.x = x
        self.r = r
        self.max_iter = max_iter
        print("Constructor call: The matrix's row and column are: ", self.x.shape[0], self.x.shape[1], "total iteration: ", self.max_iter)
        self.h = h_init
        self.errors = np.zeros(self.max_iter)

    def frobenius_norm(self):
        """ Euclidean error between x and h * h.T """

        if hasattr(self, 'h'):  # if it has attributes w and h
            error = LA.norm(self.x - self.h*self.h.T)
        else:
            error = None
        return error

    #  L2-norm with Nesterov's Optimal Gradient Method
    def nest_solver(self, beta= 1E-3):
        alpha1=1
        h_prev = self.h 
        # alpha = 0.05 / (2*L)
        grad = self.grad(self.x, self.h)

        # H=Z;    % Initialization
        # Grad=WtW*Z-WtV+beta*Z*Lp;     % Gradient
        # alpha1=1;

        for iter in range(self.max_iter):
            self.errors[iter] = LA.norm(self.x - self.h * self.h.T)  # record error
            # print(self.errors[iter])
            h0 = h_prev
            h_prev = self.proj_to_positive(self.h - beta * grad)            
            alpha2 = 0.5 * (1 + math.sqrt(1 + 4 * alpha1 * alpha1))
            print("stepsize: ", (alpha1 - 1) / alpha2)
            self.h = h_prev + ((alpha1 - 1) / alpha2) * (h_prev - h0)
            alpha1 = alpha2
            grad = self.grad(self.x, self.h)
        return self.h

    def grad(self,x,h):
        return 4 * (h * h.T * h - x * h)

    def get_error_trend(self):
        return self.errors

    def proj_to_positive(self, matrix):
        i = 0
        count = 0
        while i < matrix.shape[0]:
            j = 0
            while j < matrix.shape[1]:
                if matrix[i,j] < 0:
                    matrix[i,j] = 0
                    count += 1
                j += 1
            i += 1
        # print(count)
        return matrix


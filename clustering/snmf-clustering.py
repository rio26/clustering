import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import utils, SNMF
import numpy as np


##  ALL datasets: 
##  2 clusters: 'happy', 'be2', 'hm', 'hm2'
##  3 clusters: 'sp', 'be3', 'dc3', 'ds3'
##  4 clusters: 'g4', 'ds4'
##  5 clusters: 'ds5'
##  6 clusters: 'tar'
data_sets_2d = ['be2', 'hm', 'hm2', 'sp', 'happy', 'be3', 'dc3', 'ds3', 'g4', 'ds4', 'ds5',  'tar']
data_sets_hd = ['ch',  'hm', 'sp']

plt.figure()
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.05, top=0.98, wspace=0.2,
                    hspace=0.3)

n_cluster = 2
# for i in range(len(data_sets_2d)):
flag = True
for i in range(2):
	if i == 3 or i == 8 or i == 10:
		n_cluster += 1 # manually increasing cluster number
	X0 = np.asmatrix(utils.load_dot_mat('data/DB.mat', 'DB/' + data_sets_2d[i]))
	X = X0 * X0.T
	# print(type(X))
	initial_h = np.asmatrix(np.random.rand(X.shape[0], n_cluster))  
	cluster = SNMF.SNMF(X, h_init = initial_h, r = n_cluster, max_iter =10)
	print("Staring error: ",cluster.frobenius_norm())
	cluster_result = cluster.nest_solver()
	error = cluster.get_error_trend()
	plt.plot(error)
	plt.show()
	if flag:
		break
	# y_pred = clustering.labels_
	plt.subplot(3,4,i+1)
	plt.scatter(X[:, 0], X[:, 1],  c=y_pred, s = 5)

plt.tight_layout()
plt.show()

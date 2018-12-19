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

LABEL_COLOR_MAP = {0 : 'y',
                   1 : 'b',
                   2 : 'c',
                   3 : 'k',
                   4 : 'm',
                   5 : 'r'}

n_cluster = 2
flag = True
for i in range(len(data_sets_2d)):
# for i in range(2):
	if i == 3 or i == 8 or i == 10:
		n_cluster += 1 # manually increasing cluster number
	X0 = np.asmatrix(utils.load_dot_mat('data/DB.mat', 'DB/' + data_sets_2d[i]))
	min_diff = 0 - X0.min()
	# print(X0.shape[0], X0.shape[1], "min_diff:", min_diff)
	if min_diff > 1:
		print("Target matrix has negative element(s). Setting them positive... \n")
		for row in range(X0.shape[0]):
			for col in range(X0.shape[1]):
				# print("r", row, "c", col)
				X0[row,col] = X0[row,col] + min_diff
	# print("Line43: min_diff:", X0.min())
	print("Running on dataset:", i, " with cluster number: ", n_cluster, "...")
	X = X0 * X0.T
	# print(X)
	# print(type(X))
	initial_h = np.asmatrix(np.random.rand(X.shape[0], n_cluster))  
	# initial_h = np.asmatrix(np.random.randint(0,X.max(),size=[X.shape[0], n_cluster]))

	cluster = SNMF.SNMF(X, h_init = initial_h, r = n_cluster, max_iter =1000)
	print("Staring error: ",cluster.frobenius_norm())
	# cluster_result = cluster.proj_solver()
	# cluster_result = cluster.proj_solver_bug()
	cluster_result = cluster.mur()


	error = cluster.get_error_trend()
	# plt.plot(error)
	# print(error)
	print("Final error: ",cluster.frobenius_norm(), "Task ", i, " done. \n")
	# print(cluster_result[0,:])
	y_pred =  np.zeros([X.shape[0]])
	for row in range(len(y_pred)):
		y_pred[row] = np.argmax(cluster_result[row,:])
		# print(y_pred[row])
	# print("type & value:", type(y_pred), y_pred.shape)
	label_color = [LABEL_COLOR_MAP[l] for l in y_pred]

	plt.subplot(3,4,i+1)
	plt.scatter([X0[:, 0]], [X0[:, 1]],  c=label_color, s = 5)

plt.tight_layout()
plt.show()

import pickle
import numpy as np
data = np.load('Decompositoppn/datasets/data_demo.npy')

from pyinform.transferentropy import transfer_entropy
# TE = transfer_entropy(data[2, :], data[3, :], k=2)
result = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
	for j in range(data.shape[0]):

		X = data[i, :]
		Y = data[j, :]

		TE = transfer_entropy(X, Y, k=7)
		result[i, j] = TE

result_with_selfloop = result + np.eye(data.shape[0])
result_with_selfloop[result_with_selfloop < 0.06] = 0
# print(result_with_selfloop)

with open('causality_graph.pkl', 'wb') as f:
    pickle.dump(result_with_selfloop, f)
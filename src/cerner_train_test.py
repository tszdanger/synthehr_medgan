import os
import sys
import numpy as np
import pandas as pd
import _pickle as pickle

if __name__ == "__main__":
	train_file = "../raw/x_train_v3.npy"
	test_file = "../raw/x_test_v3.npy"

	train = np.load(train_file, allow_pickle = True)
	test = np.load(test_file, allow_pickle = True)

	train = train.reshape(train.shape[0], -1)
	test = test.reshape(test.shape[0], -1)

	print(train.shape)
	print(test.shape)

	pickle.dump(train, open("../data/cerner_train.matrix", 'wb'), -1)
	pickle.dump(test, open("../data/cerner_test.matrix", 'wb'), -1)

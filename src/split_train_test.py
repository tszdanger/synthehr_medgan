import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import _pickle as pickle

def trainTestSplit(X, y, ratio = 0.2):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = 42)
	return X_train, X_test, y_train, y_test

if __name__ == "__main__":

	matrixpath = sys.argv[1]
	savepath = sys.argv[2]
	dataset = sys.argv[3]

	if dataset == "mimic":
		matrix = matrixpath + ".matrix"
		pidpath = matrixpath + ".pids"
		headerpath = matrixpath + ".types"

	else:
		matrix = matrixpath + ".matrix"
		headerpath = matrixpath + ".types"


	ratio = 0.2

	X = np.load(matrix, allow_pickle = True)
	y = range(X.shape[0])	
	X_train, X_test, y_train, y_test = trainTestSplit(X, y, ratio)

	print(X_train.shape)
	print(X_test.shape)

	trainpath = savepath + "_train.matrix"
	testpath = savepath + "_test.matrix"
	pickle.dump(X_train, open(trainpath, 'wb'), -1)
	pickle.dump(X_test, open(testpath, 'wb'), -1)

	if dataset == "mimic":
		pidtrain =savepath + "_train.pids"
		pidtest = savepath + "_test.pids"
		copyfile(pidpath, pidtrain)
		copyfile(pidpath, pidtest)


	headertrain =savepath + "_train.types"
	headertest = savepath + "_test.types"
	copyfile(headerpath, headertrain)
	copyfile(headerpath, headertest)

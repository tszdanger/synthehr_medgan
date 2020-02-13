import sys
import h5py
import numpy as np
from datetime import datetime
import pandas as pd
import multiprocessing
import pickle
import argparse

def combineArrays(arr1, arr2):
	return np.vstack(arr1, arr2)


if __name__ == "__main__":

	# readfile = sys.argv[1] 
	# outfile = sys.argv[2]
	# reshape = sys.argv[3]
	# stack = sys.argv[4]


	train_file = "../raw/x_train_v3.npy"
	test_file = "../raw/x_test_v3.npy"
	header_file = "../raw/headers.txt"
	outfile = "../data/cerner"

	train = np.load(train_file, allow_pickle = "True")
	test = np.load(test_file, allow_pickle = "True")

	train = train.reshape(dat.shape[0], -1)	
	test = test.reshape(dat.shape[0], -1)	

	dat = combineArrays(train, test)

	# fn = readfile
	# dat = np.load(fn, allow_pickle = True)



	headers = []
	with open(header_file) as f:
		line = f.readline()
		cnt = 1
		while line:
			headers.append(line.strip())
			line = f.readline()
			cnt =+ 1

		f.close()


	headers1 = [x + "_1" for x in headers]
	headers2 = [x + "_2" for x in headers]
	headers1.extend(headers2)
	headers_all = headers1

	data = pd.DataFrame(data = dat, columns = headers_all)
	counts = data.sum(axis = 0, skipna = True)
	indexes = counts.index.tolist()
	values = counts.values.tolist()
	types = dict(zip(indexes, values))

	pickle.dump(dat, open(outfile+'.matrix', 'wb'), -1)
	pickle.dump(types, open(outfile+'.types', 'wb'), -1)


	# n = np.load("../pretrain/x_train.matrix", allow_pickle = True)
	# for line in n:
	# 	print(np.unique(line))

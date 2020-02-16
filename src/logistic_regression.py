import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


# from sklearn.metrics import 

def logisticRegressionClassification(train_mat, test_mat, headers, binary = False):
	train = pd.DataFrame(data = train_mat, columns = headers)
	test = pd.DataFrame(data = test_mat, columns = headers)

	if binary:
		train[train >= 0.5] = 1
		train[train < 0.5] = 0

		test[test >= 0.5] = 1
		test[test < 0.5] = 0

	ret_list = []
	count = 0

	for col in headers:
		count = count + 1
		print(count)

		x_train = train.drop([col], axis = 1)
		y_train = train.loc[:, col]
		# y_train[y_train >= 0.5] = 1
		# y_train[y_train < 0.5] = 0

		x_test = test.drop([col], axis = 1)
		y_test = test.loc[:, col]

		lr = LogisticRegression(max_iter = 300, random_state = 0)
		# print("Starting training")
		try:
			lr.fit(x_train, y_train)
			# print("Ending Training")
			y_pred = lr.predict(x_test)
			f1 = f1_score(y_test, y_pred)
			acc = accuracy_score(y_test, y_pred)
			recall = recall_score(y_test, y_pred)
			prec = precision_score(y_test, y_pred)

			prob_true = sum(y_test)/len(y_test)
			prob_pred = sum(y_pred)/len(y_pred)

		except:
			f1 = 0.0
			acc = 0.0
			recall = 0.0
			prec = 0.0
			prob_pred = 0.0
			prob_true = 0.0
			print("value error detected. one class of values encountered.")

		print(f1)


		retl = [col, f1, acc, recall, prec, prob_true, prob_pred]
		ret_list.append(retl)
		# x_test = train.drop([col], axis = 1)
		# y_train = train.loc[:, col]
		# rf = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
		# if count == 5:
		# 	break


	retdf = pd.DataFrame(ret_list, columns = ['variable', "f1", "accuracy", "recall", "precision", "prob_occurence_true", "prob_occurence_pred"])
	
	return retdf




if __name__ == "__main__":

	model = sys.argv[1]
	dataset = sys.argv[2]
	mode = ""
	if dataset == "cerner":
		mode = ""
	elif dataset = "mimic_na:
		mode = ""
	else:
		mode = "_binary"

	headers_dict = np.load("../converted_raw/"+ dataset + mode + ".types", allow_pickle = True)
	bh = list(headers_dict.keys())
	# bh = bh[:10]

	filename_generated = os.path.join("..", "results", dataset, model, "binary/outputs/generated.npy") 
	file_generated = np.load(filename_generated)
	# file_generated = file_generated[:10, :10]
	print(file_generated.shape)
		
	filename_test = os.path.join("../data/"+ dataset + mode + "_test.matrix")
	file_test = np.load(filename_test, allow_pickle = True)
	# file_test = file_test[:10, :10]
	print(file_test.shape)

	filename_original = os.path.join("../data/"+ dataset + mode + "_train.matrix")
	file_original = np.load(filename_original, allow_pickle = True)
	# file_original = file_original[:10, :10]	
	print(file_original.shape)



	df = logisticRegressionClassification(train_mat = file_generated, test_mat = file_test, headers = bh, binary = True)
	df.to_csv(os.path.join("..", "results", dataset, model, ''.join(["binary/sumstats/logistic_regression_metrics_", model, "_generated.csv"])), index = False)


	df = logisticRegressionClassification(train_mat = file_original, test_mat = file_test, headers = bh, binary = True)
	df.to_csv(os.path.join("..", "results", dataset, model, ''.join(["binary/sumstats/logistic_regression_metrics_", model, "_original.csv"])), index = False)


	# df = randomForestUndersampling(train_mat = file_generated, test_mat = file_test, headers = bh, binary = True)
	# df.to_csv("../summary_stats/random_forest_metrics_undersampling_randomundersampler.csv", index = False)

	# df = randomForestOversampling(train_mat = file_generated, test_mat = file_test, headers = bh, binary = True)
	# df.to_csv("../summary_stats/random_forest_metrics_oversampling.csv", index = False)


	# print(file_test.shape)


############Test




# def randomForestUndersampling(train_mat, test_mat, headers, binary = False):
# 	train = pd.DataFrame(data = train_mat, columns = headers)
# 	test = pd.DataFrame(data = test_mat, columns = headers)

# 	if binary:
# 		train[train >= 0.5] = 1
# 		train[train < 0.5] = 0

# 		test[test >= 0.5] = 1
# 		test[test < 0.5] = 0

# 	ret_list = []
# 	count = 0

# 	for col in headers:
# 		count = count + 1
# 		print(count)

# 		x_train = train.drop([col], axis = 1)
# 		y_train = train.loc[:, col]

# 		# cc = ClusterCentroids(random_state = 0)
# 		cc = RandomUnderSampler(random_state=0)
# 		# cc = NearMiss(version=1)
# 		# print("clustering......")
# 		x_train_resampled, y_train_resampled = cc.fit_resample(x_train, y_train)
# 		# print("clustered.......")

# 		x_test = test.drop([col], axis = 1)
# 		y_test = test.loc[:, col]

# 		rf = RandomForestClassifier(n_estimators = 300, random_state = 0, n_jobs = -1)
# 		# print("Starting training")
# 		rf.fit(x_train_resampled, y_train_resampled)
# 		# print("Ending Training")
# 		y_pred = rf.predict(x_test)
# 		# exit(0)

# 		f1 = f1_score(y_test, y_pred)
# 		acc = accuracy_score(y_test, y_pred)
# 		recall = recall_score(y_test, y_pred)

# 		prob_true = sum(y_test)/len(y_test)
# 		prob_pred = sum(y_pred)/len(y_pred)

# 		print(f1, pd.Series(y_test).isin([1]).sum(), pd.Series(y_pred).isin([1]).sum())

# 		retl = [col, acc, f1, recall, prob_true, prob_pred]
# 		ret_list.append(retl)
# 		# x_test = train.drop([col], axis = 1)
# 		# y_train = train.loc[:, col]
# 		# rf = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
# 		# if count == 5:
# 		# 	break


# 	retdf = pd.DataFrame(ret_list, columns = ['variable', "f1", "accuracy", "recall", "prob_occurence_true", "prob_occurence_pred"])
	
# 	return retdf




# def randomForestOversampling(train_mat, test_mat, headers, binary = False):
# 	train = pd.DataFrame(data = train_mat, columns = headers)
# 	test = pd.DataFrame(data = test_mat, columns = headers)

# 	if binary:
# 		train[train >= 0.5] = 1
# 		train[train < 0.5] = 0

# 		test[test >= 0.5] = 1
# 		test[test < 0.5] = 0

# 	ret_list = []
# 	count = 0

# 	for col in headers:
# 		count = count + 1
# 		print(count)

# 		col = headers[25]

# 		x_train = train.drop([col], axis = 1)
# 		y_train = train.loc[:, col]

# 		# ros = RandomOverSampler(random_state=0)
# 		ros = SMOTE(random_state=0)
# 		# ros = ADASYN(random_state=0)
# 		# print("generating ADASYN......")
# 		x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)
# 		# print("generated ADASYN.......")


# 		x_test = test.drop([col], axis = 1)
# 		y_test = test.loc[:, col]

# 		rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
# 		# print("Starting training")
# 		rf.fit(x_train_resampled, y_train_resampled)
# 		# print("Ending Training")
# 		y_pred = rf.predict(x_test)
# 		print(pd.Series(y_test).isin([1]).sum())
# 		print(pd.Series(y_pred).isin([1]).sum())
# 		exit(0)

# 		f1 = f1_score(y_test, y_pred)
# 		acc = accuracy_score(y_test, y_pred)
# 		recall = recall_score(y_test, y_pred)

# 		prob_true = sum(y_test)/len(y_test)
# 		prob_pred = sum(y_pred)/len(y_pred)

# 		print(f1)

# 		retl = [col, acc, f1, recall, prob_true, prob_pred]
# 		ret_list.append(retl)
# 		# x_test = train.drop([col], axis = 1)
# 		# y_train = train.loc[:, col]
# 		# rf = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
# 		# if count == 5:
# 		# 	break


# 	retdf = pd.DataFrame(ret_list, columns = ['variable', "f1", "accuracy", "recall", "prob_occurence_true", "prob_occurence_pred"])
	
# 	return retdf

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotProb(mx, my, metric, save_path, y_name):

	df_x = mx.loc[:, metric]
	df_y = my.loc[:, metric]

	# print(df_x.head(20))
	# print(df_y.head(20))
	
	
	# prob = prob.sort_values(ascending = True)

	graph = plt.figure()
	plt.scatter(x = df_x, y = df_y, s = 3, color = "red", alpha=0.8)
	# plt.plot([0, max(df['probabilities_real'])], [0, max(df['probabilities_generated'])], color = 'blue')
	plt.plot([0, max(df_x)], [0, max(df_y)], color = 'blue')
	# plt.plot([0, 1], [0, 1], color = 'blue')
	plt.title(metric + ' Comparison ' + y_name + ' vs real')
	plt.xlabel('Real ' + metric)
	plt.ylabel(y_name + ' ' + metric)
	# plt.show()
	graph.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
	
	fx = sys.argv[1] 
	fy = sys.argv[2]
	metric = sys.argv[3]
	save_path = sys.argv[4]
	y_name = sys.argv[5]

	mx = pd.read_csv(fx)
	my = pd.read_csv(fy)


	plotProb(mx, my, metric, save_path, y_name)
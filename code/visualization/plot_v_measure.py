# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path, sys


def plot_v_measure(df_iid, clustering_score):
	df_iid.v_measure.hist()
	ax = plt.gca()
	# ax.axvline(x=clustering_score, color='r', linestyle='-', label = 'DP')
	iid_mean = df_iid.v_measure.mean()
	print 'mean', ('%0.6f' % iid_mean)
	print 'max', ('%0.6f' % df_iid.v_measure.max())
	print 'sd', ('%0.6f' % df_iid.v_measure.std())
	# ax.axvline(x=iid_mean, color='b', linestyle='-', label = 'iid_mean')
	plt.xlabel('V-measure')
	plt.ylabel('# of i.i.d samples')
	plt.show()



if __name__ == '__main__':
	clustering_path = sys.argv[1]
	iid_path = sys.argv[2]

	df_iid = pd.read_csv(iid_path)

	with open(clustering_path, 'r') as f:
		clustering_score = np.float64(f.readlines()[0].strip().split('V-measure: ')[1])

	plot_v_measure(df_iid, clustering_score)
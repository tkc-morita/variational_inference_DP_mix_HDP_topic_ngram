# coding: utf-8

import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import sys, os.path


def main_loop(labels_true, num_samples):
	v_measures = []
	label_inventory = list(set(labels_true))
	data_size = len(labels_true)
	for iter_ix in range(num_samples):
		labels_pred = get_random_labels(label_inventory, data_size)
		v_measures.append(sklm.v_measure_score(labels_true, labels_pred))
	return v_measures


def get_random_labels(label_inventory, data_size):
	return np.random.choice(label_inventory, size=data_size)



if __name__ == '__main__':
	data_path = sys.argv[1]

	df_data = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	kanji2ix=dict([(u'和', 0), (u'漢', 1), (u'外', 2), (u'混', 3), (u'固', 4), (u'記号', 5)])
	df_data['actual_sublex']=df_data.wType.map(kanji2ix)
	df_data = df_data[df_data.actual_sublex.isin(range(3))] # Focus only Native, SJ, and Foreign.

	num_samples = int(sys.argv[2])

	df_result = pd.DataFrame()
	df_result['v_measure'] = main_loop(df_data.actual_sublex.tolist(), num_samples)
	df_result.to_csv('Japanese_random_baseline_%i-samples.csv' % num_samples)
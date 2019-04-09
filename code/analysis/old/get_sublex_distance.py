# coding: utf-8

import pandas as pd
import numpy as np
import itertools, sys, os.path

def get_distance_between_sublexica(df_classification):
	sublexica = [sublex for sublex in df_classification.columns.tolist() if 'sublex_' in sublex]
	distances = []
	for sublex1, sublex2 in itertools.combinations(sublexica, 2):
		prob1_list = df_classification[sublex1].tolist()
		prob2_list = df_classification[sublex2].tolist()
		dist = np.sum([
				weighted_conditional_entropy_dist(prob1, prob2)
				for prob1, prob2
					in zip(prob1_list,prob2_list)
				]) / df_classification.shape[0]
		distances.append([sublex1,sublex2,dist])
	return pd.DataFrame(distances, columns = ['sublex_a','sublex_b','distance'])



def weighted_conditional_entropy_dist(prob1, prob2):
	"""
	Get distance between two probabilities
	by weighted conditional entropy.
	"""
	return 1 - xlogx(prob1 + prob2) + xlogx(prob1) + xlogx(prob2)

def xlogx(x):
	return x * np.log(x)


if __name__ == '__main__':
	classification_path = sys.argv[1]
	result_dir = os.path.split(classification_path)[0]

	df_classification = pd.read_csv(classification_path)

	df_dist = get_distance_between_sublexica(df_classification)
	df_dist.to_csv(os.path.join(result_dir, 'sublexicon_distance.csv'), index=False)
# coding: utf-8

import pandas as pd
import sys, os.path

def count_active_sublex(df_var_bound, directory):
	for row in df_var_bound.itertuples():
		assignment_path = os.path.join(directory,row.directory,'SubLexica_assignment.csv')
		if os.path.isfile(assignment_path):
			df_assignment = pd.read_csv(assignment_path)
			num_sublex = df_assignment.most_probable_sublexicon.value_counts().size
		else:
			num_sublex = None
		df_var_bound.loc[row.Index, 'num_sublex'] = num_sublex

if __name__ == '__main__':
	directory = sys.argv[1]
	filename = os.path.join(directory, 'final_var_bounds.csv')
	df_var_bound = pd.read_csv(filename)
	count_active_sublex(df_var_bound, directory)
	df_var_bound.sort_values('final_var_bound', inplace=True)
	df_var_bound.to_csv(os.path.join(directory, 'final_var_bounds.csv'), index=False)
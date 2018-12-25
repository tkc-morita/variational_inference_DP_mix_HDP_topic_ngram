# encoding: utf-8

import numpy as np
import pandas as pd
import os.path

def get_log_sublex_assignment_probs(result_dir, ix = 'new'):
	if ix == 'new':
		df_stick = pd.read_hdf(os.path.join(result_dir, 'variational_parameters.h5'), key='/sublex/stick')
		df_stick = df_stick.sort_values('cluster_id')
		df_stick['beta_sum'] = df_stick.beta_par1 + df_stick.beta_par2
		df_stick['log_stop_prob'] = np.log(df_stick.beta_par1) - np.log(df_stick.beta_sum)
		df_stick['log_pass_prob'] = np.log(df_stick.beta_par2) - np.log(df_stick.beta_sum)
		log_assignment_probs = []
		log_cum_pass_prob = 0
		for row_tuple in df_stick.itertuples():
			log_assignment_probs.append(row_tuple.log_stop_prob + log_cum_pass_prob)
			log_cum_pass_prob += row_tuple.log_pass_prob
		log_assignment_probs.append(log_cum_pass_prob)
		log_assignment_probs = np.array(log_assignment_probs)
		return log_assignment_probs
	else:
		df_assignment = pd.read_csv(os.path.join(result_dir, 'SubLexica_assignment.csv'))
		sublex_columns = [col for col in df_assignment.columns.tolist() if col.startswith('sublex_')]
		return df_assignment.loc[ix, sublex_columns].values
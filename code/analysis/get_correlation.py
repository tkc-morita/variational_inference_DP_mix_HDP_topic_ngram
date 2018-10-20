# coding: utf-8

import pandas as pd
import sys

def get_correlation(df, var1, var2, method = 'spearman'):
	print df.loc[:,[var1,var2]].corr(method = method)

if __name__ == '__main__':
	filepath = sys.argv[1]

	df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
	experimenter = 'MoretonAmano1999'
	df = df[df.experimenter == experimenter]
	df['log_post_pred_prob_ratio_control_over_target'] = - df.log_post_pred_prob_ratio_target_over_control
	predictor = 'log_post_pred_prob_ratio_control_over_target'
	data = 'duration_boundary'

	# df = pd.read_csv(filepath, encoding='utf-8')
	# # print df
	# predictor = 'log_sublex_prob_ratio'
	# # predictor = 'log_pred_prob_ratio'
	# data = 'mean_response_diff'
	get_correlation(df, predictor, data, method = 'spearman')
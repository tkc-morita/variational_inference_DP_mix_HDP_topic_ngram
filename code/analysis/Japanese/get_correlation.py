# coding: utf-8

import pandas as pd
import argparse

def get_correlation(df, var1, var2, method = 'spearman'):
	print df.loc[:,[var1,var2]].corr(method = method)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str)
	parser.add_argument('-m','--corr_method', type=str, default='spearman')
	args = parser.parse_args()

	df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')
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
	get_correlation(df, predictor, data, method = args.corr_method)
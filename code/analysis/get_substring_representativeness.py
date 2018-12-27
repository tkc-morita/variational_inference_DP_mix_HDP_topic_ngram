# coding: utf-8

import numpy as np
import pandas as pd
import scipy.misc as spm
import encode_decode, parse_vi
import argparse, os.path

def get_log_assignment_over_others(log_assignment_probs):
	"""
	Returns a square matrix A s.t.:
	a_ij = -np.inf if i==j
	a_ij = p(t_j) / (1 - p(t_i)) otherwise
	"""
	log_assignment_to_others = np.repeat(log_assignment_probs[np.newaxis,:], log_assignment_probs.size, axis=0)
	np.fill_diagonal(log_assignment_to_others, -np.inf)
	log_assignment_to_others = spm.logsumexp(log_assignment_to_others, axis=1)
	log_assignment_over_others = log_assignment_probs[np.newaxis,:] - log_assignment_to_others[:,np.newaxis]
	np.fill_diagonal(log_assignment_over_others, -np.inf)
	return log_assignment_over_others

def get_representativeness(log_like, log_assignment_over_others):
	"""
	Tenenbaum & Griffiths (2001: Eq.4).
	"""
	log_denominator = spm.logsumexp(log_like[np.newaxis,:] + log_assignment_over_others, axis=1)
	return log_like - log_denominator

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('likelihood_csv', type=str, help='Path to the csv file containing likelihood info.')
	parser.add_argument('result_dir', type=str, help='Path to the directory file containing learning results.')
	parser.add_argument('conditioning_string_csv', type=str, help='Substrings to condition the target probability. Each symbols should be comma-separated.')
	args = parser.parse_args()

	df_code = pd.read_csv(os.path.join(args.result_dir, 'symbol_coding.csv'), encoding='utf-8')
	encoder,decoder = encode_decode.df2coder(df_code)

	conditioning_string_coded = [encoder[symbol] for symbol in args.conditioning_string_csv.decode('utf-8').split(',')]

	df_like = pd.read_csv(args.likelihood_csv, encoding='utf-8')
	sub_df_like = df_like
	string_cols = sorted([col for col in df_like.columns.tolist() if col.startswith('symbol_')])
	for pos_col,symbol in zip(string_cols[::-1],conditioning_string_coded[::-1]):
		sub_df_like = sub_df_like[sub_df_like[pos_col]==symbol]
	log_like = np.ma.log(sub_df_like.groupby('sublex', sort=True).prob.sum().values)

	log_assignment_probs = parse_vi.get_log_sublex_assignment_probs(args.result_dir)
	log_assignment_over_others = get_log_assignment_over_others(log_assignment_probs)

	representativeness = get_representativeness(log_like, log_assignment_over_others)

	for sublex,r in enumerate(representativeness):
		print('sublex_{sublex}: {r:.4f}'.format(sublex=sublex, r=r))
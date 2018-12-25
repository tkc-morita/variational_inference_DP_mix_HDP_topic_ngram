# coding: utf-8

import numpy as np
import pandas as pd
import scipy.misc as spm
import encode_decode, parse_vi
import argparse, os.path


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
	log_like = np.log(sub_df_like.groupby('sublex', sort=True).prob.sum().values)

	log_assignment_probs = parse_vi.get_log_sublex_assignment_probs(args.result_dir)

	log_posterior = log_like + log_assignment_probs
	log_posterior -= spm.logsumexp(log_posterior)

	for sublex,p in enumerate(np.exp(log_posterior)):
		print('sublex_{sublex}: {p:.4f}'.format(sublex=sublex, p=p))
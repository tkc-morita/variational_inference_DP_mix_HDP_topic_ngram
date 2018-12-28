# coding: utf-8

import numpy as np
import pandas as pd
import scipy.misc as spm
import encode_decode, parse_vi
import get_substring_representativeness as rep
import argparse, os.path

def to_int_code(string, encoder):
	try:
		out = int(string)
	except:
		out = encoder[string]
	return out

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('likelihood_csv', type=str, help='Path to the csv file containing likelihood info.')
	parser.add_argument('result_dir', type=str, help='Path to the directory file containing learning results.')
	parser.add_argument('string_length', type=int, help='Length of substrings to rank.')
	parser.add_argument('top_k', type=int, help='Length of the ranking.')
	parser.add_argument('-f','--frequency_csv', type=str, default=None, help='Path to the csv file containing frequency info. If specified, the ranking will be limited to substrings with positive frequency in the file.')
	args = parser.parse_args()

	df_code = pd.read_csv(os.path.join(args.result_dir, 'symbol_coding.csv'), encoding='utf-8')
	encoder,decoder = encode_decode.df2coder(df_code)


	df_like = pd.read_csv(args.likelihood_csv, encoding='utf-8')
	string_cols = sorted([col for col in df_like.columns.tolist() if col.startswith('symbol_')])[-args.string_length:]
	df_like = df_like.groupby(string_cols+['sublex']).sum().reset_index()
	df_like['log_like'] = df_like.prob.map(np.ma.log)

	if not args.frequency_csv is None:
		df_freq = pd.read_csv(args.frequency_csv, encoding='utf-8')
		df_freq = df_freq.rename(columns={'value':string_cols[-1],'sublex_id':'sublex'})
		df_freq[string_cols[-1]] = df_freq[string_cols[-1]].map(lambda s: to_int_code(s, encoder))
		if args.string_length > 1:
			df_freq = pd.concat([
						df_freq,
						df_freq.context.str.split('_', expand=True).rename(columns={ix:col for ix,col in enumerate(string_cols[:-1])}).applymap(lambda s: to_int_code(s, encoder))
						],
						axis=1)

		df_like = pd.merge(df_like, df_freq, on=string_cols+['sublex'])
		df_like = df_like[df_like.frequency > 0]

	log_assignment_probs = parse_vi.get_log_sublex_assignment_probs(args.result_dir)
	log_assignment_over_others = rep.get_log_assignment_over_others(log_assignment_probs)

	for substring,sub_df_like in df_like.groupby(string_cols):
		sub_df_like = sub_df_like.sort_values('sublex')
		score = rep.get_representativeness(sub_df_like.log_like.values, log_assignment_over_others)
		df_like.loc[sub_df_like.index, 'representativeness'] = score

	for sublex, sub_df_like in df_like.groupby('sublex'):
		sub_df_like = sub_df_like.sort_values('representativeness', ascending=False).head(args.top_k)
		sub_df_like = sub_df_like.reset_index(drop=True)
		sub_df_like.loc[:,'rank'] = sub_df_like.index + 1
		substring_csv = sub_df_like[string_cols[0]].map(decoder)
		for col in string_cols[1:]:
			substring_csv = substring_csv + ',' + sub_df_like[col].map(decoder)
		sub_df_like.loc[:,'substring_csv'] = substring_csv
		if args.frequency_csv is None:
			filename = 'representative_substrings_length-{length}_of_sublex-{sublex}_top_{top_k}.tsv'.format(length=args.string_length, sublex=sublex, top_k=args.top_k)
		else:
			filename = 'representative_substrings_WITH-NON-ZERO-FREQ_length-{length}_of_sublex-{sublex}_top_{top_k}.tsv'.format(length=args.string_length, sublex=sublex, top_k=args.top_k)
		sub_df_like.to_csv(os.path.join(args.result_dir, filename), sep='\t', encoding='utf-8')

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

def reformat_symbols(symbol):
	return symbol.replace('r',u'ɾ').replace(u'ä','a').replace('9',u'ɰ̃')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('likelihood_csv', type=str, help='Path to the csv file containing likelihood info.')
	parser.add_argument('result_dir', type=str, help='Path to the directory containing learning results.')
	parser.add_argument('frequency_csv', type=str, help='Path to the csv file containing frequency info.')
	parser.add_argument('string_length', type=int, help='Length of substrings to rank.')
	args = parser.parse_args()

	df_code = pd.read_csv(os.path.join(args.result_dir, 'symbol_coding.csv'), encoding='utf-8')
	encoder,decoder = encode_decode.df2coder(df_code)


	df_like = pd.read_csv(args.likelihood_csv, encoding='utf-8')
	string_cols = sorted([col for col in df_like.columns.tolist() if col.startswith('symbol_')])[-args.string_length:]
	df_like = df_like.groupby(string_cols+['sublex']).sum().reset_index()
	df_like['log_like'] = df_like.prob.map(np.ma.log)

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
	# df_like = df_like[df_like.frequency > 0]

	log_assignment_probs = parse_vi.get_log_sublex_assignment_probs(args.result_dir)
	log_assignment_over_others = rep.get_log_assignment_over_others(log_assignment_probs)

	for substring,sub_df_like in df_like.groupby(string_cols):
		sub_df_like = sub_df_like.sort_values('sublex')
		score = rep.get_representativeness(sub_df_like.log_like.values, log_assignment_over_others)
		df_like.loc[sub_df_like.index, 'representativeness'] = score

	for sublex, sub_df_like in df_like.groupby('sublex'):
		sub_df_like = sub_df_like.sort_values('representativeness', ascending=False)
		# sub_df_like = sub_df_like.reset_index(drop=True)
		sub_df_like.loc[:,'rank_in_sublex'] = np.arange(sub_df_like.shape[0]) + 1
		df_like.loc[sub_df_like.index,'rank_in_sublex'] = sub_df_like.rank_in_sublex

	substring_csv = df_like[string_cols[0]].map(lambda s: reformat_symbols(decoder[s]))
	for col in string_cols[1:]:
		substring_csv = substring_csv + ',' + df_like[col].map(lambda s: reformat_symbols(decoder[s]))
	df_like.loc[:,'substring_csv'] = substring_csv
	filename = 'representativeness_{length}-gram-substrings.tsv'.format(length=args.string_length)
	df_like.loc[:,['substring_csv','sublex','representativeness','rank_in_sublex','frequency']].to_csv(os.path.join(args.result_dir, filename), sep='\t', encoding='utf-8',index=False)

# coding: utf-8

import pandas as pd
import numpy as np
import sys, os.path, itertools

def get_pairwise_KL_divergence(df_post_ngram, context_frequency):
	# print np.max(np.abs(df_post_ngram.groupby(['sublex_id','context']).prob.sum() - 1))
	df_post_ngram = df_post_ngram.sort_values(['sublex_id','context','value'])
	df_post_ngram['log_prob'] = np.log(df_post_ngram.prob)
	df_post_ngram['neg_entropy'] = df_post_ngram.prob * df_post_ngram.log_prob
	df_results = pd.DataFrame(columns=['context','context_in_data','sublex_A','sublex_B','kl_divergence_AB','kl_divergence_BA'])
	sublex_list = sorted(df_post_ngram.sublex_id.drop_duplicates().tolist())
	for sublex_A in sublex_list[:-1]:
		df_sublex_A = df_post_ngram[df_post_ngram.sublex_id == sublex_A].reset_index(drop=True)
		# print 'A'
		# print df_sublex_A
		neg_entropy_A = df_sublex_A.groupby('context').neg_entropy.sum()
		for sublex_B in sublex_list[sublex_A+1:]:
			df_sublex_B = df_post_ngram[df_post_ngram.sublex_id == sublex_B].reset_index(drop=True)
			# print 'B'
			# print df_sublex_B
			df_sublex_B['kl_divergence_AB'] = df_sublex_A.neg_entropy - (df_sublex_A.prob * df_sublex_B.log_prob)
			df_sublex_B['kl_divergence_BA'] = df_sublex_B.neg_entropy - (df_sublex_B.prob * df_sublex_A.log_prob)
			df_results_sub = df_sublex_B.groupby('context')['kl_divergence_AB','kl_divergence_BA'].sum()
			df_results_sub['context'] = df_results_sub.index
			df_results_sub = df_results_sub.sort_values('context')
			df_results_sub['sublex_A'] = sublex_A
			df_results_sub['sublex_B'] = sublex_B
			df_results_sub['context_frequency_A'] = df_results_sub.context.map(lambda context: context_frequency[context][sublex_A])
			df_results_sub['context_frequency_B'] = df_results_sub.context.map(lambda context: context_frequency[context][sublex_B])
			df_results_sub['context_frequency_all'] = df_results_sub.context.map(lambda context: np.sum(context_frequency[context]))
			df_results_sub['context_in_data'] = df_sublex_A[df_sublex_A.value==0].sort_values('context').context_in_data.tolist()
			df_results = df_results.append(df_results_sub, ignore_index=True)
	df_results['kl_divergence_avg'] = (df_results.kl_divergence_AB + df_results.kl_divergence_BA) / 2.0
	return df_results

def code_data(csv_data_list, symbol2code):
	return [map(lambda key: symbol2code[key], string.split(',')) for string in csv_data_list]
	
def get_context_frequency(df_sublex_assignment, coded_data, start_code, n):
	df_sublex_assignment = df_sublex_assignment.loc[:,df_sublex_assignment.columns.str.startswith('sublex_')]
	num_sublex = df_sublex_assignment.shape[1]
	inventory = list(set(itertools.chain.from_iterable(coded_data)))
	inventory.append(len(inventory))
	context_frequency = {'_'.join(map(str, context_list)):np.zeros(num_sublex)
							for context_list
							in itertools.product(inventory, repeat=n-1)
							}
	mat_assignments = df_sublex_assignment.rename(
							columns={
									col_name:int(col_name.split('_')[1])
									for col_name in df_sublex_assignment.columns.tolist()
									}
									).ix[:,range(num_sublex)].values
	for coded_string,sublex_assignment in zip(coded_data, mat_assignments):
		for ngram_window in zip(*[([start_code]*(n-1)+coded_string)[i:] for i in range(n)]):
			context = '_'.join(map(str, ngram_window[:-1]))
			context_frequency[context] += sublex_assignment
	return context_frequency



if __name__ == '__main__':
	path = sys.argv[1]
	result_dir,filename = os.path.split(path)
	n = int(list(filename.split('gram')[0])[-1])
	df_post_ngram = pd.read_csv(path)
	df_sublex_assignment = pd.read_csv(os.path.join(result_dir, 'SubLexica_assignment.csv'))
	df_data = pd.read_csv('../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns.tsv', sep='\t', encoding='utf-8')
	df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	df_code.set_index('symbol', inplace=True)
	coded_data = code_data(df_data.IPA_csv.tolist(), df_code.to_dict()['code'])
	start_code = df_code.to_dict()['code']['START']
	context_frequency = get_context_frequency(df_sublex_assignment, coded_data, start_code, n)

	df_kl_div = get_pairwise_KL_divergence(
					df_post_ngram,
					context_frequency
					).sort_values('kl_divergence_avg', ascending = False)
	df_kl_div.to_csv(os.path.join(result_dir, 'kl-divergence_bw_sublex.csv'), index=False)

# coding: utf-8

import pandas as pd
import numpy as np
import sys, os.path, itertools

	
def get_ngram_frequency(mat_assignments, data, start_symbol, n):
	ngram_frequency = {}
	for string,sublex_assignment in zip(data, mat_assignments):
		for ngram_window in zip(*[([start_symbol]*(n-1)+string)[i:] for i in range(n)]):
			ngram = '_'.join(ngram_window)
			if ngram in ngram_frequency.keys():
				ngram_frequency[ngram] += sublex_assignment
			else:
				ngram_frequency[ngram] = sublex_assignment
	return ngram_frequency

def get_ngram_entropy(ngram_frequency, num_sublex):
	exp_freq_col_names = [
							'expected_frequency_in_sublex_%i' % sublex
							for sublex in xrange(num_sublex)
						]
	df_ngram_entropy = pd.DataFrame(
							columns=[
								'ngram_sequence','frequency','entropy_over_sublex','most_frequent_sublex'
								]+exp_freq_col_names
							)
	for ngram,expected_freqs  in ngram_frequency.iteritems():
		all_freq = np.sum(expected_freqs)
		sub_df = pd.DataFrame(expected_freqs[np.newaxis,:], columns=exp_freq_col_names)
		sub_df['ngram_sequence'] = ngram
		sub_df['frequency'] = all_freq
		sub_df['most_frequent_sublex'] = np.argmax(expected_freqs)
		normalized_freqs = expected_freqs / all_freq
		sub_df['entropy_over_sublex'] = -np.sum(normalized_freqs*np.log(normalized_freqs))
		df_ngram_entropy = df_ngram_entropy.append(
												sub_df,
												ignore_index=True
											)
	return df_ngram_entropy

if __name__ == '__main__':
	result_dir = sys.argv[1]
	n = int(sys.argv[2])
	df_sublex_assignment = pd.read_csv(os.path.join(result_dir, 'SubLexica_assignment.csv'))
	df_data = pd.read_csv('../data/with_superlong_vowels/BCCWJ_frequencylist_suw_ver1_0_core-nouns.tsv', sep='\t', encoding='utf-8')
	data = [string.split(',') for string in df_data.IPA_csv.tolist()]
	df_sublex_assignment = df_sublex_assignment.loc[:,df_sublex_assignment.columns.str.startswith('sublex_')]
	num_sublex = df_sublex_assignment.shape[1]
	mat_assignments = df_sublex_assignment.rename(
							columns={
									col_name:int(col_name.split('_')[1])
									for col_name in df_sublex_assignment.columns.tolist()
									}
									).ix[:,range(num_sublex)].values
	ngram_frequency = get_ngram_frequency(mat_assignments, data, 'START', n)
	df_ngram_entropy = get_ngram_entropy(ngram_frequency, num_sublex).sort_values('entropy_over_sublex')
	df_ngram_entropy.to_csv(os.path.join(result_dir, '%igram-entropy-over-sublexica.csv' % n), encoding='utf-8', index=False)

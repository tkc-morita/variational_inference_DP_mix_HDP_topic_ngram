# coding: utf-8

import pandas as pd
import numpy as np
import scipy.misc as spm
# import scipy.special as sps
import scipy.stats as spst
import sys, os.path

# def get_representativeness(
# 			df_ngram,
# 			log_assignment_over_others
# 			):
# 	df_ngram = df_ngram.sort_values('sublex_id')
# 	for (context, value), sub_df in df_ngram.groupby(['context','value']):
# 		log_ngram_prob_x_assignment = (
# 										np.log(sub_df.prob)[np.newaxis,:]
# 										+
# 										log_assignment_over_others
# 										)
# 		# np.fill_diagonal(log_ngram_prob_x_assignment, np.float64(1))
# 		log_denominator = spm.logsumexp(log_ngram_prob_x_assignment, axis=1)
# 		df_ngram.loc[
# 			(df_ngram.context == context) & (df_ngram.value == value)
# 			,
# 			'representativeness'
# 			] = np.log(sub_df.prob) - log_denominator
# 	return df_ngram

def get_ngram_frequency(mat_assignments, data, start_symbol, n, df_ngram):
	df_ngram = df_ngram.sort_values('sublex_id')
	df_ngram['frequency'] = 0
	df_ngram['expected_frequency'] = np.float64(0)
	for string,sublex_assignment in zip(data, mat_assignments):
		# print ''.join(string)
		for ngram_window in zip(*[([start_symbol]*(n-1)+string)[i:] for i in range(n)]):
			context = '_'.join(ngram_window[:-1])
			value = ngram_window[-1]
			assert not context in [u'9ː_9ː', u'9ː_b', u'9ː_k', u'9ː_nː'], ''.join(string)
			df_ngram.loc[
				(df_ngram.decoded_context == context) & (df_ngram.decoded_value == value)
				,
				'expected_frequency'
				] += sublex_assignment
			df_ngram.loc[
				(df_ngram.decoded_context == context) & (df_ngram.decoded_value == value)
				,
				'frequency'
				] += 1
	return df_ngram



if __name__ == '__main__':
	path = sys.argv[1]
	df_ngram = pd.read_csv(path)
	df_ngram = df_ngram[df_ngram.context_in_data]

	result_dir,filename = os.path.split(path)
	n = int(filename.split('gram')[0].split('_')[-1])

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

	df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	df_code.set_index('code', inplace=True)
	decoder = df_code.symbol.to_dict()

	df_ngram['decoded_value'] = df_ngram.value.map(decoder)
	df_ngram['decoded_context'] = df_ngram.context.map(lambda context: '_'.join(map(lambda code: decoder[int(code)], context.split('_'))))


	datapath = sys.argv[2]
	df_sublex_assignment = pd.read_csv(os.path.join(result_dir, 'SubLexica_assignment.csv'))
	df_data = pd.read_csv(datapath, sep='\t', encoding='utf-8')
	data = [string.split(',') for string in df_data.IPA_csv.tolist()]
	df_sublex_assignment = df_sublex_assignment.loc[:,df_sublex_assignment.columns.str.startswith('sublex_')]
	num_sublex = df_sublex_assignment.shape[1]
	mat_assignments = df_sublex_assignment.rename(
							columns={
									col_name:int(col_name.split('_')[1])
									for col_name in df_sublex_assignment.columns.tolist()
									}
									).ix[:,range(num_sublex)].values
	df_ngram = get_ngram_frequency(mat_assignments, data, 'START', n, df_ngram)
	# df_ngram = df_ngram[df_ngram.frequency > 0]


	# sublex_ids_str = sys.argv[2].split(',')
	# sublex_ids = map(int, sublex_ids_str)
	# df_ngram = df_ngram[df_ngram.sublex_id.isin(sublex_ids)]
	# log_assignment_probs = log_assignment_probs[sublex_ids]
	# num_sublex = log_assignment_probs.size
	# log_assignment_to_others = np.repeat(log_assignment_probs[np.newaxis,:], num_sublex, axis=0)
	# np.fill_diagonal(log_assignment_to_others, -np.inf)
	# log_assignment_to_others = spm.logsumexp(log_assignment_to_others, axis=1)
	# log_assignment_over_others = log_assignment_probs[np.newaxis,:] - log_assignment_to_others[:,np.newaxis]
	# np.fill_diagonal(log_assignment_over_others, -np.inf)

	# df_ngram = get_representativeness(df_ngram, log_assignment_over_others)
	df_ngram = df_ngram.sort_values('representativeness', ascending=False)

	

	df_ngram.to_csv(
		os.path.join(
			result_dir,
			'posterior_%igram_with-frequency.csv' % (n)
			)
			,
			encoding = 'utf-8',
			index = False
			)
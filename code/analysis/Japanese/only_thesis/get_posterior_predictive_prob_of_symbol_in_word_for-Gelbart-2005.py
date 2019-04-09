# coding: utf-8

import numpy as np
import pandas as pd
import scipy.misc as spm
import os, sys
import posterior_predictive_inferences as ppi
import encode_decode as edcode



def normalize_over_prefix(df, log_prob='log_prob_target'):
	for trial_name, sub_df in df.groupby('trial_name'):
		log_normalizer = spm.logsumexp(sub_df[log_prob])
		df.loc[
			df.trial_name==trial_name,
			'normalized_prob_target'
			] = np.exp(
					df.loc[
						df.trial_name==trial_name,
						log_prob
						]
					-
					log_normalizer
					)



if __name__=='__main__':

	ngram_path = sys.argv[1]
	n = int(ngram_path.split('gram')[0][-1])
	result_dir = os.path.split(ngram_path)[0]
	hdf5_path = os.path.join(result_dir, 'variational_parameters.h5')

	df_ngram = pd.read_csv(ngram_path)

	df_stick = pd.read_hdf(hdf5_path, key='/sublex/stick')
	log_assignment_probs = ppi.get_log_assignment_probs(df_stick)

	df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	encoder,decoder = edcode.df2coder(df_code)


	data_path = sys.argv[2]
	df_data = pd.read_csv(data_path, encoding='utf-8', sep='\t').fillna('')
	# prefixes = edcode.encode_data(df_data.prefix, encoder, add_end_symbol = False)
	# targets = edcode.encode_data(df_data.target_c + ',' + df_data.target_v, encoder, add_end_symbol = False)
	# suffixes = [(encoder['END'],)]*df_data.shape[0]
	# inventory = [code for value,code in encoder.iteritems() if not value in ['END', 'START']]
	# start_code = encoder['START']

	words = edcode.encode_data(df_data.word, encoder)
	start_code = encoder['START']

	# log_probs = ppi.get_log_posterior_predict_prob_of_target(prefixes, targets, suffixes, df_ngram, log_assignment_probs, n, start_code, inventory)
	unnormalized_log_probs = ppi.get_unnormalized_log_posterior_predict_prob_of_target(words, df_ngram, log_assignment_probs, n, start_code)

	# df_data['log_prob_target'] = log_probs
	df_data['unnormalized_log_prob_target'] = unnormalized_log_probs
	normalize_over_prefix(df_data, 'unnormalized_log_prob_target')

	# classification_probs = ppi.posterior_predict_classification(words, df_ngram, log_assignment_probs, n, start_code)
	# for sublex_id, class_probs in enumerate(classification_probs.T):
	# 	df_data.loc[:,'sublex_%i' % sublex_id] = class_probs

	datafile_root = os.path.splitext(os.path.split(data_path)[1])[0]
	ratio_filename = datafile_root+'_log-posterior-predictive-prob_target.tsv'

	df_data.to_csv(os.path.join(result_dir, ratio_filename), index=False, sep='\t', encoding='utf-8')
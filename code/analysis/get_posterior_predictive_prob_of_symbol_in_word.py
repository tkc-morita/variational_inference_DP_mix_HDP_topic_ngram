# coding: utf-8

import numpy as np
import pandas as pd
import scipy.misc as spm
import os, sys
import posterior_predictive_inferences as ppi
import encode_decode as edcode




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
	prefixes = edcode.encode_data(df_data.prefix, encoder, add_end_symbol = False)
	targets = edcode.encode_data(df_data.target, encoder, add_end_symbol = False)
	controls = edcode.encode_data(df_data.control, encoder, add_end_symbol = False)
	suffixes = edcode.encode_data(df_data.suffix, encoder)
	inventory = [code for value,code in encoder.iteritems() if not value in ['END', 'START']]
	start_code = encoder['START']

	log_probs = ppi.get_log_posterior_predict_prob_of_target_and_control(prefixes, targets, controls, suffixes, df_ngram, log_assignment_probs, n, start_code, inventory)
	df_log_probs = pd.DataFrame(log_probs, columns=['target_log_prob','control_log_prob'])
	log_normalizer = spm.logsumexp(log_probs, axis=1)
	df_log_probs['normalized_target_prob'] = np.exp(df_log_probs.target_log_prob - log_normalizer)
	df_log_probs['normalized_control_prob'] = np.exp(df_log_probs.control_log_prob - log_normalizer)



	datafile_root = os.path.splitext(os.path.split(data_path)[1])[0]
	ratio_filename = datafile_root+'_log-posterior-predictive-prob_target-and-control_test.tsv'
	df_log_probs = pd.concat(
						[
							df_log_probs,
							df_data
						],
						axis=1
						)

	df_log_probs.to_csv(os.path.join(result_dir, ratio_filename), index=False, sep='\t', encoding='utf-8')
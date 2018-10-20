# coding: utf-8

import numpy as np
import pandas as pd
# import scipy.misc as spm
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
	df_data = pd.read_csv(data_path, encoding='utf-8', sep='\t')
	start_code = encoder['START']


	base = edcode.encode_data(df_data.base_DISC.map(lambda x: ','.join(list(x))), encoder)
	df_data['base_log_prob'] = ppi.get_unnormalized_log_posterior_predict_prob_of_target(base, df_ngram, log_assignment_probs, n, start_code)

	sity = edcode.encode_data(df_data.sity_DISC.map(lambda x: ','.join(list(x))), encoder)
	df_data['sity_log_prob'] = ppi.get_unnormalized_log_posterior_predict_prob_of_target(sity, df_ngram, log_assignment_probs, n, start_code)

	kity = edcode.encode_data(df_data.kity_DISC.map(lambda x: ','.join(list(x))), encoder)
	df_data['kity_log_prob'] = ppi.get_unnormalized_log_posterior_predict_prob_of_target(kity, df_ngram, log_assignment_probs, n, start_code)

	ness = edcode.encode_data(df_data.ness_DISC.map(lambda x: ','.join(list(x))), encoder)
	df_data['ness_log_prob'] = ppi.get_unnormalized_log_posterior_predict_prob_of_target(ness, df_ngram, log_assignment_probs, n, start_code)


	datafile_root = os.path.splitext(os.path.split(data_path)[1])[0]
	result_filename = datafile_root+'_posterior-predictive-prob.csv'
	df_data.to_csv(os.path.join(result_dir, result_filename), index=False, encoding = 'utf-8')

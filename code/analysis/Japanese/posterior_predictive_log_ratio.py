# coding: utf-8

import pandas as pd
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
	start_code = encoder['START']



	log_ratio = ppi.posterior_predict_symbol_in_word(prefixes, targets, controls, suffixes, df_ngram, log_assignment_probs, n, start_code)
	log_ratio_prefix_MAP = ppi.posterior_predict_symbol_in_word_by_prefix_MAP(prefixes, targets, controls, suffixes, df_ngram, log_assignment_probs, n, start_code)


	datafile_root = os.path.splitext(os.path.split(data_path)[1])[0]
	ratio_filename = datafile_root+'_log-ratio-posterior-predictive-prob-target-vs-control_test.tsv'
	df_ratio = df_data.loc[:,['prefix','target','control','suffix','experimenter','duration_boundary','group_identifier','actual_sublex']]
	df_ratio.loc[:,'log_post_pred_prob_ratio_target_over_control'] = log_ratio
	df_ratio.loc[:,'log_post_pred_prob_ratio_target_over_control_prefix_MAP'] = log_ratio_prefix_MAP
	df_ratio.to_csv(os.path.join(result_dir, ratio_filename), index=False, sep='\t', encoding='utf-8')
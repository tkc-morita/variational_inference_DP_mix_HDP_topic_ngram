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
	target_data = edcode.encode_data(df_data.IPA_csv, encoder)
	start_code = encoder['START']



	classification_target = ppi.posterior_predict_classification(target_data, df_ngram, log_assignment_probs, n, start_code)

	df_classification = pd.DataFrame(classification_target, columns=[('sublex_%i' % i) for i in range(classification_target.shape[1])])
	df_classification['MAP_classification'] = np.argmax(classification_target, axis=1)
	# df_classification['katakana'] = df_data.katakana
	# df_classification['orthography'] = df_data.orthography
	# df_classification['wType'] = df_data.wType



	datafile_root = os.path.splitext(os.path.split(data_path)[1])[0]
	classification_filename = datafile_root+'_posterior-predictive-classification.csv'
	df_classification.to_csv(os.path.join(result_dir, classification_filename), index=False, encoding = 'utf-8')

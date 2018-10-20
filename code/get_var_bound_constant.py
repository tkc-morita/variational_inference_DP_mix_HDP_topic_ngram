# coding: utf-8

import variational_inference_DP_topic_ngram_mix_indep_matrix as vin
import numpy as np
import os, datetime, shutil
import pandas as pd
import argparse

def get_result_dir():
	with open('result_dir_info.txt', 'r') as f:
		result_dir = os.path.join(
						f.readlines()[0].strip(),
						'DP-mix_HDP-topic-ngram'
						)
	return result_dir

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--ngram", type=int, help="Context length of ngram", default=3)
	parser.add_argument("-d", "--data", type=str, help="Path to data", default='../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_4th-ed.tsv')
	parser.add_argument("-s", "--sublex", type=int, help="Max # of sublexica", default=10)
	parser.add_argument("-c", "--DirichletConcentration", type=np.float64, help="Concentration for top level dirichlet distribution", default=1.0)
	parser.add_argument("-k", "--dataColumn", type=str, help="Column name for the inputs.", default='IPA_csv')

	options=vars(parser.parse_args())
	
	datapath = options['data']
	
	data_prefix=os.path.split(datapath.rstrip('/'))[-1]

	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')
	
	data,encoder,decoder = vin.code_data(df[options['dataColumn']])
	
	
	T_scalar = 2 # (# of tables) = 2*(# of symbols).
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
	dirichlet_concentration = options['DirichletConcentration']

	data_root = os.path.splitext(datapath.split('/')[-1])[0]
	
	tmp_result_path = os.path.join(
					get_result_dir(),
					data_root,
					'constant-of-ELBO'
					)
	os.makedirs(tmp_result_path)

	num_sublex = options['sublex']
	n = options['ngram']
	base_num_clusters = len(decoder)*2

	vi = vin.VariationalInference(
			num_sublex,
			data,
			n,
			base_num_clusters,
			concent_priors,
			dirichlet_concentration,
			tmp_result_path
			)

	print(str(vi.get_var_constant()))






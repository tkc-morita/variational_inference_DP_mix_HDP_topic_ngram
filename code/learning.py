# coding: utf-8

import variational_inference_DP_mix_HDP_topic_ngram as vin
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
	parser.add_argument("data", type=str, help="Path to data", default='../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_5th-ed.tsv')
	parser.add_argument("-n", "--ngram", type=int, help="Ngram length", default=3)
	parser.add_argument("-i", "--iterations", type=int, help="Maxmum # of iterations", default=2500)
	parser.add_argument("-T", "--tolerance", type=np.float64, help="Tolerance level to detect convergence", default=0.1)
	parser.add_argument("-s", "--sublex", type=int, help="Max # of sublexica", default=10)
	parser.add_argument("-c", "--topic_base_counts", type=np.float64, help="Concentration for top level dirichlet distribution", default=1.0)
	parser.add_argument("-j", "--jobid", type=str, help='Job ID #', default=None)
	parser.add_argument("-k", "--data_column", type=str, help="Column name for the inputs.", default='IPA_csv')

	options=vars(parser.parse_args())
	
	datapath = options['data']
	
	data_prefix=os.path.split(datapath.rstrip('/'))[-1]

	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')
	
	data,encoder,decoder = vin.code_data(df[options['data_column']])
	
	
	T_scalar = 2 # (# of tables) = 2*(# of symbols).
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
	topic_base_counts = options['topic_base_counts']

	result_dir = get_result_dir()
	data_filename = os.path.splitext(datapath.split('/')[-1])[0]
	
	now = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
	if options["jobid"] is None:
		job_id = now
	else:
		job_id = options["jobid"]+'_'+now
	tmp_result_path = os.path.join(
					result_dir,
					data_filename,
					job_id
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
			topic_base_counts,
			tmp_result_path
			)
	
	max_iters = options['iterations']
	min_increase = options['tolerance']

	vi.train(max_iters, min_increase)
	vi.save_results(decoder)





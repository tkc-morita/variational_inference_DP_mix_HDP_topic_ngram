# coding: utf-8

import variational_inference_DP_topic_ngram_mix_indep_Poisson_length as vin
import numpy as np
import os, datetime, shutil
import pandas as pd
import argparse

def get_result_dir():
	with open('result_dir_info.txt', 'r') as f:
		result_dir = os.path.join(
						f.readlines()[0].strip(),
						'topic_indep_Poisson_backward'
						)
	return result_dir

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--ngram", type=int, help="Context length of ngram", default=3)
	parser.add_argument("-d", "--data", type=str, help="Path to data", default='../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_revised.tsv')
	parser.add_argument("-i", "--iterations", type=int, help="Maxmum # of iterations", default=2500)
	parser.add_argument("-T", "--threshold", type=np.float64, help="Threshold to detect convergence", default=0.1)
	parser.add_argument("-s", "--sublex", type=int, help="Max # of sublexica", default=10)
	parser.add_argument("-c", "--DirichletConcentration", type=np.float64, help="Concentration for top level dirichlet distribution", default=1.0)
	parser.add_argument("-j", "--jobid", type=str, help='Job ID #', default=None)

	options=vars(parser.parse_args())
	
	datapath = options['data']
	
	data_prefix=os.path.split(datapath.rstrip('/'))[-1]

	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')
	
	data,encoder,decoder = vin.code_data(df.IPA_csv)

	data = [word[::-1] for word in data] # Model samples segments in the backward way.
	
	
	T_scalar = 2 # (# of tables) = 2*(# of symbols).
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
	dirichlet_concentration = options['DirichletConcentration']

	result_dir = get_result_dir()
	data_root = os.path.splitext(datapath.split('/')[-1])[0]
	
	now = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
	if options["jobid"] is None:
		job_id = now
	else:
		job_id = options["jobid"]+'_'+now
	tmp_result_path = os.path.join(
					result_dir,
					data_root,
					("Dirichlet-concent-%s" % str(dirichlet_concentration)),
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
			dirichlet_concentration,
			tmp_result_path
			)
	
	max_iters = options['iterations']
	min_increase = options['threshold']

	vi.train(max_iters, min_increase)
	vi.save_results(decoder)

	# result_path = os.path.join(
	# 				persistent_dir,
	# 				data_root,
	# 				now
	# 				)
	# shutil.copytree(tmp_result_path, result_path)



# coding: utf-8

import variational_inference_DP_mix_HDP_topic_ngram as vin
import numpy as np
import os, datetime, shutil
import pandas as pd
import argparse


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("data_path", type=str, help="Path to data")
	parser.add_argument("-r", "--result_path", type=str, help="Path to the directory where you want to save results. (Several subdirectories will be created.)", default='../results_debug')
	parser.add_argument("-n", "--ngram", type=int, help="Ngram length. Only 2 and longer grams are currently supported (i.e., no support for 1gram).", default=3)
	parser.add_argument("-i", "--iterations", type=int, help="Maxmum # of iterations", default=2500)
	parser.add_argument("-T", "--tolerance", type=np.float64, help="Tolerance level to detect convergence", default=0.1)
	parser.add_argument("-s", "--sublex", type=int, help="Max # of sublexica", default=10)
	parser.add_argument("-c", "--topic_base_counts", type=np.float64, help="Concentration for top level dirichlet distribution", default=1.0)
	parser.add_argument("-j", "--jobid", type=str, help='Job ID #', default=None)
	parser.add_argument("-k", "--data_column", type=str, help="Column name for the inputs.", default='IPA_csv')
	parser.add_argument("-S", "--shape_of_sublex_concentration", type=np.float64, help="Shape parameter of the Gamma prior on the concentration of the sublexicon DP.", default=10.0)
	parser.add_argument("-R", "--rate_of_sublex_concentration", type=np.float64, help="Rate (= inverse of scale) parameter of the Gamma prior on the concentration of the sublexicon DP.", default=10.0)
	parser.add_argument("--max_segment_clusters", type=int, default=None, help="Variational upper bound on # of segment clusters (tables). = 2*(num_symbols+2) by default")


	options=vars(parser.parse_args())
	
	
	data_prefix=os.path.split(options['data_path'].rstrip('/'))[-1]

	df = pd.read_csv(options['data_path'], sep='\t', encoding='utf-8')
	
	data,encoder,decoder = vin.code_data(df[options['data_column']])
	
	
	T_scalar = 2 # (# of tables) = 2*(# of symbols).
	concent_priors = np.array((options['shape_of_sublex_concentration'],options['rate_of_sublex_concentration'])) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
	topic_base_counts = options['topic_base_counts']

	result_dir = os.path.join(
						options['result_path'],
						'DP-mix_HDP-topic-ngram'
						)
	data_filename = os.path.splitext(options['data_path'].split('/')[-1])[0]
	
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
	assert n > 1, "Only 2 and longer grams are supported."

	if options['max_segment_clusters'] is None:
		max_segment_clusters = len(decoder)*2
	else:
		max_segment_clusters = options['max_segment_clusters']

	vi = vin.VariationalInference(
			num_sublex,
			data,
			n,
			max_segment_clusters,
			concent_priors,
			topic_base_counts,
			tmp_result_path
			)
	
	max_iters = options['iterations']
	min_increase = options['tolerance']

	vi.train(max_iters, min_increase)
	vi.save_results(decoder)





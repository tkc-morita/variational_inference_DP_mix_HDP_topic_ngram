# coding: utf-8

import variational_inference_DP_ngram_mix_indep_no_label as vin
import numpy as np
import os,datetime,sys,argparse
import pandas as pd

def get_result_dir():
	with open('result_dir_info.txt') as f:
		dir=f.readlines()[0].strip()
	return dir

def main_loop(
		training_data,
		n,
		sl_T,
		T_base,
		concent_priors,
		max_iters,
		min_increase,
		decoder,
		data_prefix,
		test_data,
		):
# 	data_fileroot=os.path.splitext(datapath.split('/')[-1])[0]
	now = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
	result_path=os.path.join(
						get_result_dir(),
						data_prefix+('_train-and-test_with_%i-gram_indep_base' % n),
						now
						)
	os.makedirs(result_path)
	vi = vin.VariationalInference(
					sl_T,
					training_data,
					n,
					T_base,
					concent_priors,
					result_path
					)
	vi.train(max_iters, min_increase)
	
	df_post_pred=pd.DataFrame()
	df_post_pred['log_pred_prob']=vi.get_log_posterior_pred(test_data)
	df_post_pred['test_data_id']=df_post_pred.index
	df_post_pred.to_csv(os.path.join(result_path,'log_pred_probs.csv'), index=False)

	vi.save_results(decoder)
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-n",
		"--ngram",
		default=3,
		type=int,
		help="Context length of ngram"
		)
	parser.add_argument(
		"-d",
		"--data",
		default="../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_170712180028",
		type=str,
		help="Path to data"
		)
	parser.add_argument(
		"-i",
		"--iterations",
		default=10000,
		type=int,
		help="Maxmum # of iterations"
		)
	parser.add_argument(
		"-T",
		"--threshold",
		default=0.01,
		type=np.float64,
		help="Threshold to detect convergence"
		)
	parser.add_argument(
		"-s",
		"--sublex",
		default=10,
		type=int,
		help="Max # of sublexica"
		)
	options=parser.parse_args()
	
	datapath = options.data
	data_prefix=datapath.split('/')[-1]
	df_train = pd.read_csv(os.path.join(datapath,data_prefix+'_training_data.tsv'),
						sep='\t',
						encoding='utf-8'
						)
	df_test = pd.read_csv(os.path.join(datapath,data_prefix+'_test_data.tsv'),
						sep='\t',
						encoding='utf-8'
						)
	training_data,test_data,encoder,decoder = vin.code_data(df_train.IPA_csv,test_data=df_test.IPA_csv)
	
	n = options.ngram
	# T_scalar = 2 # (# of tables) = 2*(# of symbols).
	sl_T=options.sublex
	T_base=len(decoder)*2
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
	max_iters = options.iterations
	min_increase = options.threshold
	start = datetime.datetime.now()
	main_loop(
		training_data,
		n,
		sl_T,
		T_base,
		concent_priors,
		max_iters,
		min_increase,
		decoder,
		data_prefix,
		test_data
		)
	print 'Time spent',str(datetime.datetime.now()-start)
	

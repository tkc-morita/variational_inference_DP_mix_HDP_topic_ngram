# coding: utf-8

import variational_inference_DP_ngram_mix as vinm
import sys,os,datetime
import pandas as pd
import numpy as np

if __name__=='__main__':
# 	warnings.simplefilter('error', UserWarning)
	datapath = sys.argv[1]
	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')
# 	str_data = list(df.IPA_)
# 	with open(datapath,'r') as f: # Read the phrase file.
# 		str_data = [phrase.replace('\r','\n').strip('\n').split('\t')[0]
# 								for phrase in f.readlines()
# 								]
	customers,decoder = vinm.code_data(list(df.IPA_csv.astype(str)))
	now = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
# 	result_path = ('/om/user/tmorita/results/'+os.path.splitext(datapath.split('/')[-1])[0]+'_'+now+'/')
	result_path = ('./results/'+os.path.splitext(datapath.split('/')[-1])[0]+'_'+now+'/')
	os.makedirs(result_path)
	sl_T = int(sys.argv[2])
	n = int(sys.argv[3])
	T_base = len(decoder)*2 # Number of symbols x 2
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
# 	noise = np.float64(sys.argv[4])
	max_iters = int(sys.argv[4])
	min_increase = np.float64(sys.argv[5])
	start = datetime.datetime.now()
	vinm.make_variational_inference(
			sl_T,
			customers,
			n,
			T_base,
			concent_priors,
			result_path,
			max_iters,
			min_increase,
			decoder
		)
	print 'Time spent',str(datetime.datetime.now()-start)
# coding: utf-8

import pandas as pd
import numpy as np
import variational_inference_DP_topic_ngram_mix_indep_mora as vin
import sys

if __name__ == '__main__':
	datapath = sys.argv[1]
	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')

	coded_data,encoder,decoder,segment2mora = vin.code_data(df.IPA_csv)
	print 'Max moraic length'
	print np.max([np.sum([segment2mora[segment] for segment in word]) for word in coded_data])
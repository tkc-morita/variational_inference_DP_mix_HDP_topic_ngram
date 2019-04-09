# coding: utf-8

import numpy as np
import pandas as pd
# import scipy.misc as spm
import os, sys
import posterior_predictive_inferences as ppi
import encode_decode as edcode






if __name__=='__main__':

	assignment_path = sys.argv[1]

	df_assign = pd.read_csv(assignment_path)

	data_path = sys.argv[2]
	df_data = pd.read_csv(data_path, encoding='utf-8', sep='\t')

	df_assign.loc[:,'orthography'] = df_data.lemma
	df_assign.loc[:,'DISC'] = df_data.DISC

	lemma_path = sys.argv[3]
	df_lemma = pd.read_csv(lemma_path, encoding='utf-8', sep='\t')


	df_lemma = pd.merge(df_lemma, df_assign, how='left', on='orthography')
	df_lemma = df_lemma.drop(columns='customer_id').drop_duplicates()


	result_dir = os.path.split(assignment_path)[0]
	datafile_root = os.path.splitext(os.path.split(lemma_path)[1])[0]
	classification_filename = datafile_root+'_posterior-classification.tsv'
	df_lemma.to_csv(os.path.join(result_dir, classification_filename), index=False, encoding = 'utf-8', sep='\t')

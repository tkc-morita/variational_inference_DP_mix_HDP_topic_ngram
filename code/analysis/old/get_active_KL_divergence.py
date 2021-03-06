# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path



if __name__ == '__main__':
	path = sys.argv[1]
	data_path = '../BCCWJ_frequencylist_suw_ver1_0_core-nouns.tsv'
	sublex_ids_str = sys.argv[2].split(',')
	sublex_ids = map(int, sublex_ids_str)
	df = pd.read_csv(path)
	df = df[df.sublex_A.isin(sublex_ids) & df.sublex_B.isin(sublex_ids)].sort_values('kl_divergence_avg', ascending = False)
	result_dir = os.path.split(path)[0]
	# df.to_csv(os.path.join(result_dir, 'kl_divergence_bw_%s.csv' % '-'.join(sublex_ids_str)))
	result_filepath_png = os.path.join(result_dir, 'hist_kl_divergence_bw_%s.png' % '-'.join(sublex_ids_str))
	hist_KL_divergence(df, result_filepath_png)
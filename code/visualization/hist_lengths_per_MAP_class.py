# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path

def draw_histogram(df, result_path):
	for map_sublex, df_map_sublex in df.groupby(by='most_probable_sublexicon'):
		df_map_sublex.length.hist(bins=np.arange(20)-0.5)
		# plt.xlim((0,20))
		plt.xticks(np.arange(20))
		plt.title('Segmental lengths of words MAP-classified into %s (N=%i)' % (map_sublex,df_map_sublex.shape[0]))
		plt.xlabel('(Segmental) word length')
		plt.ylabel('Frequency')
		plt.savefig(os.path.join(result_path, 'hist_lengths_MAP-%s.png' % (map_sublex)))
		plt.gcf().clear()


def get_lengths(df):
	return df.IPA_csv.apply(lambda string: len(string.split(',')))

if __name__ == '__main__':
	result_path = sys.argv[1]
	# true_sublex = sys.argv[2]
	
	df_result = pd.read_csv(os.path.join(result_path, 'SubLexica_assignment.csv'))
	df_data = pd.read_csv('../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_revised.tsv', sep='\t', encoding='utf-8')
	# kanji2alph=dict([(u'漢', 'SJ'), (u'和','Native'), (u'混', 'Mixed'), (u'外', 'Foreign'), (u'固', 'Proper'), (u'記号', 'Symbols')])
	# df_result['actual_sublex']=df_data.wType.map(kanji2alph)

	df_result['length'] = get_lengths(df_data)

	draw_histogram(df_result, result_path)

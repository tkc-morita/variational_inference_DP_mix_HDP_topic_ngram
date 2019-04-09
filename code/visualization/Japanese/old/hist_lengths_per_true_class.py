# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path

def draw_histogram(df):
	for sublex, df_sublex in df.groupby(by='actual_sublex'):
		df_sublex.length.hist(bins=np.arange(20)-0.5)
		# plt.xlim((0,20))
		plt.xticks(np.arange(20))
		plt.title('Histogram of word lengths in %s (N=%i)' % (sublex,df_sublex.shape[0]))
		plt.savefig(os.path.join('../data', 'hist_lengths_true-%s.png' % (sublex)))
		plt.gcf().clear()


def get_lengths(df):
	return df.IPA_csv.apply(lambda string: len(string.split(',')))

if __name__ == '__main__':
	
	df_data = pd.read_csv('../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_revised.tsv', sep='\t', encoding='utf-8')
	kanji2alph=dict([(u'漢', 'SJ'), (u'和','Native'), (u'混', 'Mixed'), (u'外', 'Foreign'), (u'固', 'Proper'), (u'記号', 'Symbols')])
	df_data['actual_sublex']=df_data.wType.map(kanji2alph)

	df_data['length'] = get_lengths(df_data)

	draw_histogram(df_data)

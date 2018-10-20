# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path

def draw_histogram(df, result_path):
	for sublex, df_sublex in df.groupby(by='sublex_id'):
		df_sublex.plot.bar(x='length', y='cum_prob', legend=False, width=1)
		# plt.xlim((0,20))
		plt.title('Posterior predictive segmental lengths of words in sublex_%s' % (sublex))
		plt.ylabel('Posterior predictive probability')
		plt.savefig(os.path.join(result_path, 'posterior-predictive-ngram-lengths_%s.png' % (sublex)))
		plt.gcf().clear()


def get_lengths(df):
	return df.IPA_csv.apply(lambda string: len(string.split(',')))

if __name__ == '__main__':
	result_path = sys.argv[1]
	# true_sublex = sys.argv[2]
	
	df_result = pd.read_csv(os.path.join(result_path, 'posterior_length_probs.csv'))
	# df_data = pd.read_csv('../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_revised.tsv', sep='\t', encoding='utf-8')
	# kanji2alph=dict([(u'漢', 'SJ'), (u'和','Native'), (u'混', 'Mixed'), (u'外', 'Foreign'), (u'固', 'Proper'), (u'記号', 'Symbols')])
	# df_result['actual_sublex']=df_data.wType.map(kanji2alph)

	# df_result['length'] = get_lengths(df_data)

	draw_histogram(df_result, result_path)

# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path
import seaborn as sns

def plot_ngram_representativeness(df_ngram, result_dir, n):
	sns.violinplot(data=df_ngram, x='sublex_id', y='representativeness')
	plt.title('Violin plot of %i-gram representativeness.' % n)
	plt.savefig(os.path.join(result_dir, '%igram_representativeness.png' % n))

if __name__ == '__main__':
	result_path = sys.argv[1]
	result_dir, filename = os.path.split(result_path)
	n = int(filename.split('gram')[0][-1])
	# sublex_id = int(sys.argv[2])

	# list_length = int(sys.argv[3])
	
	df_ngram = pd.read_csv(result_path)

	plot_ngram_representativeness(df_ngram, result_dir, n)
	# df_data = pd.read_csv('../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_revised.tsv', sep='\t', encoding='utf-8')
	# kanji2alph=dict([(u'漢', 'SJ'), (u'和','Native'), (u'混', 'Mixed'), (u'外', 'Foreign'), (u'固', 'Proper'), (u'記号', 'Symbols')])
	# df_result['actual_sublex']=df_data.wType.map(kanji2alph

	# df_result['length'] = get_lengths(df_data)

	# df_ngram = df_ngram[df_ngram.sublex_id == sublex_id]
	# df_ngram = df_ngram.sort_values('representativeness', ascending=False)
	# df_ngram = df_ngram.head(n=list_length)
	# df_ngram.to_csv(os.path.join(result_dir, 'top-%i-representative_ngrams.csv' % list_length))



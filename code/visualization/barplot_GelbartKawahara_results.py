# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os.path

def barplot(df, x, y, result_path):
	sns.set_style("whitegrid")
	ax = sns.barplot(x=x, y=y, data=df)
	plt.setp(ax.get_xticklabels(), rotation=45)
	plt.ylim((-8,8))
	plt.tight_layout()
	plt.savefig(result_path, bbox_inches='tight')

if __name__ == '__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, encoding='utf-8')

	ordered_categories = [u'saɾada-haɾada', u'kɯɾabɯ-naɾabɯ', u'nabi-tabi', u'medaɾɯ-kɯdaɾɯ', u'maɡɯ-toɡɯ', u'neɡa-saɡa']
	# ordered_categories = [u'nasa-mosa', u'sahaɾa-nohaɾa']
	df.loc[:,'word_pair'] = pd.Categorical(df.word_pair,
									categories=ordered_categories
									)

	result_dir = os.path.split(data_path)[0]
	result_path = os.path.join(result_dir, 'GelbartKawahara2007_mean_responce_diff_C.png')

	barplot(df, 'word_pair', 'mean_response_diff', result_path)
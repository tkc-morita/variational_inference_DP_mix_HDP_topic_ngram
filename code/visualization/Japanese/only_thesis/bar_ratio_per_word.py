# coding: utf-8

import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
# import my_autopct

def stacked_barplot(df, result_dir):
	
	sublex_ids=[colname for colname in df.columns if colname.startswith('sublex_')]

	df_summed = df.set_index('IPA').loc[:,sublex_ids]
	# df_summed = df.iloc[:,sublex_ids]
	fig = plt.figure(figsize = (8.0,8.0 / 3))
	ax = plt.subplot(111)
	df_summed.plot.barh(stacked=True, ax=ax)
	plt.gca().invert_yaxis()
	# plt.title("Posterior predictive probability of words' classification to sublexica.")
	plt.title("Posterior predictive probability of prefixes' classification to sublexica.")
	plt.xlabel('Sublexical classification probability')
	# plt.ylabel('Word')
	plt.ylabel('prefix')
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.tight_layout()
	# plt.savefig(os.path.join(result_dir,'bar_per-word_V.png'), bbox_inches='tight')
	plt.savefig(os.path.join(result_dir,'prefix_classification.png'), bbox_inches='tight')
	# plt.show()
	plt.gcf().clear()
	
	
	
if __name__=='__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, sep=',', encoding='utf-8')

	# Gelbart & Kawahara (2007)
	# df['IPA'] = df.IPA.str.replace(u'ä',u'a').str.replace(u'r',u'ɾ').str.replace(u'g',u'ɡ')
	# df = df[df.experimenter == 'GelbartKawahara2007']
	# df = df[df.stimulus_type == 'control']
	# df = df[df.group_identifier >= 6]

	# Gelbart (2005) prefix
	df['IPA'] = df.prefix.str.replace(u'ä',u'a')


	result_path = os.path.split(data_path)[0]
	# if not os.path.isdir(result_path):
	# 	os.makedirs(result_path)


	stacked_barplot(
		df,
		result_path
		)
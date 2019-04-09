# coding: utf-8

import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
# import my_autopct

def stacked_barplot(df, result_dir):
	
	sublex_ids=[colname for colname in df.columns if colname.startswith('sublex_')]

	df_summed = df.set_index('orthography').loc[:,sublex_ids]
	print(df_summed.shape[0])
	# df_summed = df_summed.iloc[20:,:] #df_summed.head(n = df_summed.shape[0] / 3)
	# print(df_summed.shape[0])
	fig = plt.figure(figsize = (8.0, df_summed.shape[0] / 3.0))
	# fig = plt.figure(figsize = (4.0, 19 / 3.0))
	ax = plt.subplot(111)
	df_summed.plot.barh(stacked=True, ax=ax)
	plt.gca().invert_yaxis()
	ax.axvline(x=0.5, color = 'm')
	plt.title("Posterior probability of dative verbs' classification to sublexica.")
	plt.xlabel('Sublexical classification probability')
	plt.ylabel('Word')
	legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.xticks([0.0,0.5,1.0])
	# legend.remove()
	plt.tight_layout()
	plt.savefig(os.path.join(result_dir,'bar_per-word_Latin_grammatical_whole.png'), bbox_inches='tight')
	# plt.savefig(os.path.join(result_dir,'FOR-PRESENTATION_bar_per-word_Latin_grammatical_2.png'), bbox_inches='tight')
	# plt.show()
	# plt.gcf().clear()
	
	
	
if __name__=='__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')

	df = df[~pd.isnull(df).any(axis=1)]
	
	linguistic_work = 'Levin1993_Latin'
	# linguistic_work = 'YangMontrul2017_Latin'
	# linguistic_work = 'YangMontrul2017_none'
	my_etym = 'non_Latin'
	# df = df[df.linguistic_work == linguistic_work]
	df = df[df.linguistic_work != linguistic_work]
	# df = df[df.MyEtym == my_etym]
	df = df[df.MyEtym != my_etym]
	df = df.sort_values('orthography')

	result_path = os.path.splitext(data_path)[0]
	if not os.path.isdir(result_path):
		os.makedirs(result_path)


	stacked_barplot(
		df,
		result_path
		)
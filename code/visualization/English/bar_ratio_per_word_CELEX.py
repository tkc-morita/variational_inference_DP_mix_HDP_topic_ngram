# coding: utf-8

import pandas as pd
import sys, os
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import seaborn as sns
# import my_autopct

def stacked_barplot(df, result_dir):
	
	# sublex_ids=[colname for colname in df.columns if colname.startswith('sublex_')]
	sublex_ids=['sublex_0','sublex_2','sublex_5']

	df_summed = df.set_index('lemma').loc[:,sublex_ids].rename(columns={col:format_sublex_name(col) for col in df.columns.tolist() if col.startswith('sublex_')})
	# print(df_summed.shape[0])
	df_summed = df_summed.iloc[35:,:] #df_summed.head(n = df_summed.shape[0] / 3)
	print(df_summed.shape[0])
	fig = plt.figure(figsize = (8.0, df_summed.shape[0] / 3.0))
	# fig = plt.figure(figsize = (4.0, 19 / 3.0))
	ax = plt.subplot(111)
	df_summed.plot.barh(stacked=True, ax=ax)
	plt.gca().invert_yaxis()
	ax.axvline(x=0.5, color = 'gray', linestyle='--')
	plt.title("Posterior probability of dative verbs' classification.")
	plt.xlabel('Classification probability')
	plt.ylabel('Word')
	legend = ax.legend(loc='upper right')
	# legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	# handles, labels = ax.get_legend_handles_labels()
	# fig.legend(handles, labels, loc='upper right')
	# plt.xticks([0.0,0.5,1.0])
	# legend.remove()
	plt.tight_layout()
	# plt.savefig(os.path.join(result_dir,'bar_per-word_Latin_grammatical_whole.png'), bbox_inches='tight')
	# plt.savefig(os.path.join(result_dir,'bar_per-word_Latin_ungrammatical_whole.png'), bbox_inches='tight')
	# plt.savefig(os.path.join(result_dir,'bar_per-word_non-Latin_grammatical_2.png'), bbox_inches='tight')
	plt.savefig(os.path.join(result_dir,'bar_per-word_non-Latin_2.png'), bbox_inches='tight')
	# plt.savefig(os.path.join(result_dir,'FOR-PRESENTATION_bar_per-word_Latin_grammatical_2.png'), bbox_inches='tight')
	# plt.show()
	# plt.gcf().clear()
	
def format_sublex_name(original):
	ix = int(original.split('_')[-1])
	ix2new_name = {0:'-ability',2:'Latinate',5:'Germanic'}
	if ix in ix2new_name:
		return r'\textsc{Sublex}\textsubscript{$\approx$' + ix2new_name[ix] + r'}'
	else:
		return original
	
if __name__=='__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')

	df = df[~df.most_probable_sublexicon.isnull()]
	
	# df = df[df.double_object == 'grammatical']
	# df = df[df.double_object != 'grammatical']
	# df = df[df.MyEtym == 'Latin']
	df = df[df.MyEtym == 'non_Latin']
	# print(df)
	df = df.sort_values('lemma')

	result_path = os.path.splitext(data_path)[0]
	if not os.path.isdir(result_path):
		os.makedirs(result_path)


	stacked_barplot(
		df,
		result_path
		)
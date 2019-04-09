# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)  
# plt.rc('font', family='serif')
import argparse, os.path

def heatmap(df_result, result_dir, row_group, col_group, vmax = None, fmt='.2g'):
	mat = get_matrix(df_result, row_group, col_group)
	ax_hm = sns.heatmap(mat, cmap='binary', annot=True, fmt=fmt, vmax=vmax)
	ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0)
	if not vmax is None:
		ax_cb = plt.gcf().axes[-1]
		ax_cb.set_yticklabels(list(ax_cb.get_yticklabels()[:-1]) + ['$\geq{vmax}$'.format(vmax=vmax)])
	plt.xlabel(col_group)
	# plt.show()
	plt.savefig(os.path.join(result_dir, 'heatmap.png'), bbox_inches='tight')


def get_matrix(df_result, row_group, col_group):
	return pd.pivot_table(df_result, values='cardinality', index=row_group, columns=col_group, fill_value=0)

def format_sublex_name(sublex_name):
	return (r'\textsc{Sublex}\textsubscript{$\approx$%s}' % sublex_name)

def rename_sublex(sublex_name):
	ix = int(sublex_name.split('_')[1])
	ix2name = {0:'Foreign', 3:'SJ', 4:'Native', 5:'Symbols'}
	return ix2name[ix]

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('result_path', type=str, help='Path to the classification results.')
	parser.add_argument('data_path', type=str, help='Path to the data, containing the grand truth classification info.')
	args = parser.parse_args()

	df_result = pd.read_csv(args.result_path)
	# columns = [colname for colname in df_result.columns if colname.startswith('sublex_')]
	# df_result = df_result.loc[:,columns].rename(columns = {colname:format_sublex_name(colname) for colname in columns})

	result_dir = os.path.split(args.result_path)[0]
	
	df_data=pd.read_csv(args.data_path, sep='\t', encoding='utf-8')
	kanji2alph=dict([(u'漢', 'SJ'), (u'和','Native'), (u'混', 'Mixed'), (u'外', 'Foreign'), (u'固', 'Proper'), (u'記号', 'Symbols')])
	df_result['Etymological sublexicon']=pd.Categorical(df_data.wType.map(kanji2alph), ['Native', 'SJ', 'Foreign', 'Symbols', 'Mixed', 'Proper'])

	df_result = df_result.groupby('Etymological sublexicon').most_probable_sublexicon.value_counts().to_frame('cardinality').reset_index()
	df_result.loc[:,'most_probable_sublexicon'] = pd.Categorical(
													df_result.most_probable_sublexicon.map(rename_sublex).map(format_sublex_name),
													[format_sublex_name(sublex) for sublex in ['Native', 'SJ', 'Foreign', 'Symbols']]
													)
	df_result = df_result.rename(columns={'most_probable_sublexicon':'Predicted word categories'})

	heatmap(
		df_result,
		result_dir,
		'Etymological sublexicon',
		'Predicted word categories',
		vmax = 3000,
		fmt='d'
		)
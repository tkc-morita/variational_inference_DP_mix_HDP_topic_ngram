# coding: utf-8

# import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)  
# plt.rc('font', family='serif')
import argparse, os.path
import heatmap_class_alignment as hca

def format_sublex_name(sublex_name):
	return (r'\textsc{Sublex}\textsubscript{$\approx$%s}' % sublex_name)
	# return (r'\textsc{Sublex}\textsubscript{%s}' % sublex_name)

def rename_sublex(sublex_name):
	ix = int(sublex_name.split('_')[1])
	ix2name = {0:'-ability', 2:'Latinate', 5:'Germanic'}
	return ix2name[ix]
	# return ix

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('result_path', type=str, help='Path to the classification results.')
	parser.add_argument('data_path', type=str, help='Path to the data, containing the grand truth classification info.')
	args = parser.parse_args()

	df_result = pd.read_csv(args.result_path)
	# columns = [colname for colname in df_result.columns if colname.startswith('sublex_')]
	# df_result = df_result.loc[:,columns].rename(columns = {colname:format_sublex_name(colname) for colname in columns})

	result_dir = os.path.split(args.result_path)[0]
	
	df_data = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')
	df_data = df_data[~df_data.origin.isnull()]
	df_result = df_result.loc[df_data.index,:]
	grand_truth_label = 'Etymological origin'
	df_result[grand_truth_label]=pd.Categorical(df_data.origin.str.replace('|','/'), ['AngloSaxon', 'OldNorse', 'Dutch', 'AngloSaxon/OldNorse', 'AngloSaxon/Dutch', 'Latin', 'French', 'French/Latin'])#, 'LatinatesOfGermanic', 'AMBIGUOUS'])
	# df_result = df_result[df_result['Etymological sublexicon']!='__NA__']
	# df_result.loc[:,'Etymological sublexicon'] = df_result['Etymological sublexicon'].str.replace('_','-')
	# df_result = df_result[df_result['Etymological sublexicon'].isin(['Germanic', 'Latinate', 'Greek', 'Celtic', 'Balto_Slavic', 'Indo_Iranian'])]


	# MAP
	df_result = df_result.groupby(grand_truth_label).most_probable_sublexicon.value_counts().to_frame('cardinality').reset_index()
	df_result.loc[:,'most_probable_sublexicon'] = pd.Categorical(
													df_result.most_probable_sublexicon.map(rename_sublex).map(format_sublex_name),
													# [format_sublex_name(sublex) for sublex in ['Native', 'SJ', 'Foreign', 'Symbols']]
													)
	predicted_category_label = 'Predicted word categories'
	df_result = df_result.rename(columns={'most_probable_sublexicon':predicted_category_label})

	# Sum of classification probability
	# sublex_cols = [col for col in df_result.columns.tolist()if col.startswith('sublex_')]
	# df_result = df_result.groupby('Etymological sublexicon')[sublex_cols].sum(axis=0).reset_index()
	# df_result = df_result.melt(id_vars=['Etymological sublexicon'], value_vars=sublex_cols, var_name='Predicted word categories', value_name='cardinality')
	# df_result.loc[:,'Predicted word categories'] = df_result['Predicted word categories'].map(rename_sublex).map(format_sublex_name)

	hca.heatmap(
		df_result,
		result_dir,
		grand_truth_label,
		predicted_category_label,
		vmax=3000,
		fmt='d'
		)
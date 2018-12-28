# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from mpl_toolkits.mplot3d import Axes3D
import argparse, os.path
import heatmap_class_alignment as heat

def plot_3dbars(df, result_dir, xcol, ycol):
	zcol = 'cardinality'

	datasize = df.shape[0]
	width = 0.8
	dx = [width] * datasize
	dy = [width] * datasize
	dz = df[zcol]

	x2pos = {value:pos for pos,value in enumerate(reversed(df[xcol].drop_duplicates().tolist()), start=0)}
	y2pos = {value:pos for pos,value in enumerate(df[ycol].drop_duplicates().tolist(), start=0)}
	
	xpos = df[xcol].map(x2pos)
	ypos = df[ycol].map(y2pos)
	zpos = [0] * datasize


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='y')

	ax.set_xlabel(xcol)
	ax.set_ylabel(ycol)
	ax.set_zlabel(zcol)

	ax.set_xticks(np.array([pos for value,pos in x2pos.items()]) + 0.5*width)
	ax.set_xticklabels([value for value,pos in x2pos.items()])
	ax.set_yticks(np.array([pos for value,pos in y2pos.items()]) + 0.5*width)
	ax.set_yticklabels([value for value,pos in y2pos.items()])


	plt.show()
	# plt.savefig(os.path.join(result_dir, '3Dbar.png'), bbox_inches="tight")


def format_sublex_name(sublex_name):
	return (r'\textsc{Sublex\textsubscript{%s}}' % sublex_name.split('_')[1])

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
	df_result['Etymological sublexicon']=pd.Categorical(df_data.wType.map(kanji2alph), ['Foreign', 'SJ', 'Native', 'Symbols', 'Mixed', 'Proper'])

	df_result = df_result.groupby('Etymological sublexicon').most_probable_sublexicon.value_counts().to_frame('cardinality').reset_index()
	df_result.loc[:,'most_probable_sublexicon'] = df_result.most_probable_sublexicon.map(format_sublex_name)
	df_result = df_result.rename(columns={'most_probable_sublexicon':'Predicted word categories'})

	plot_3dbars(
		df_result,
		result_dir,
		'Etymological sublexicon',
		'Predicted word categories'
		)
# coding: utf-8

import pandas as pd
import argparse, os.path
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from cycler import cycler
c = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=c)
import seaborn as sns
# import my_autopct

def bar(result_path):
	df=pd.read_csv(result_path)
	result_dir = os.path.dirname(result_path)
	sublex_ids=[colname for colname in df.columns if colname.startswith('sublex_')]
	# sns.countplot(x='most_probable_sublexicon', data=df)
	# sns.set_style("white")
	# plt.ylim(ymax=ymax)

	df_sublex = df.loc[:,sublex_ids]

	# df_sublex.sum(axis=0).plot.pie(autopct=my_autopct.my_autopct, legend=True)
	# plt.ylabel('')
	# plt.title('Sublexical distribution of words (N=%i)' % df.shape[0])
	# plt.savefig(os.path.join(result_dir,'bar_whole.png'), bbox_inches='tight')
	# plt.gcf().clear()

	map_counts = df.most_probable_sublexicon.value_counts(sort=False)
	num_categories = map_counts.size
	# cm = {format_sublex_name(sublex):'C{ix}'.format(ix=ix) for ix,sublex in enumerate(map_counts.index.tolist())}
	# plt.style.use('seaborn')
	ax = map_counts.to_frame().T.rename(columns=format_sublex_name).plot.barh(stacked=True)#, colormap=lambda x: 'C{ix}'.format(ix=int(x*(num_categories-1))))
	for p in ax.patches:
		width = p.get_width()
		height = p.get_height()
		x,y = p.get_xy()
		width_ratio = width/float(map_counts.sum())
		if width_ratio>0.2:
			ax.annotate('{width} ({percent:.1f}'.format(width=width, percent=(100.0*width_ratio)) + r'\%)', (x+width_ratio*width*0.5, y+.5*height))
	# map_counts.reset_index().melt(value_vars=['most_probable_sublexicon'], var_name='MAP categories', value_name='Counts').plot.barh(stacked=True)
	# plt.ylabel('')
	ax.set_yticklabels([''])
	plt.xlabel('Cardinality')
	plt.title('MAP classification of words (N=%i)' % df.shape[0])
	plt.savefig(os.path.join(result_dir,'bar_whole_MAP.png'), bbox_inches='tight')
	plt.gcf().clear()

def to_tab10(x):
	print(x)
	return 'C{ix}'.format(ix=int(x))


def format_sublex_name(original):
	ix = int(original.split('_')[-1])
	ix2new_name = {0:'-ability',2:'Latinate',5:'Germanic'}
	if ix in ix2new_name:
		return r'\textsc{Sublex}\textsubscript{$\approx$' + ix2new_name[ix] + r'}'
	else:
		return original
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('result_path', type=str, help='Path to the csv containing classification results.')
	args = parser.parse_args()
	bar(args.result_path)
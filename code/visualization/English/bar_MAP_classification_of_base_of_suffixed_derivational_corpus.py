# coding: utf-8

import numpy as np
import pandas as pd
import os.path, argparse, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../analysis'))
import posterior_predictive_inferences as ppi
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from cycler import cycler
c = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=c)
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap,BoundaryNorm
# import seaborn as sns
# import scipy.stats as sstat
import rpy2.robjects
from rpy2.robjects.packages import importr
emt = importr('EMT')
# import my_autopct

def bar(df_data, df_origin, result_dir, normalize=True, multinom_ps=None, all_sublex=False):
	sublex_ids = df_data.map_sublex_base.unique().tolist()

	# Suffix loop
	df_suffixed = pd.DataFrame(columns = sublex_ids)
	for suffix,sub_df in df_data.groupby('last_suffix'):
		df_suffixed.loc[suffix,sublex_ids] = sub_df.map_sublex_base.value_counts(normalize=normalize).reindex(sublex_ids, fill_value=0)
		df_suffixed.loc[suffix,'count'] = sub_df.shape[0]

	df_origin = df_origin.set_index('suffix')
	df_suffixed = pd.merge(df_suffixed, df_origin.loc[:,['MyOrigin']].rename(columns={'MyOrigin':'origin'}), how='left', left_index=True, right_index=True)
	df_suffixed['origin'] = pd.Categorical(df_suffixed.origin, categories=['Germanic','?','Latinate'], ordered=True)
	df_suffixed = df_suffixed.sort_values(['origin','count'], ascending=[True,False])
	df_suffixed = df_suffixed.drop(columns='count')
	df_suffixed = df_suffixed[sorted([col for col in df_suffixed.columns.tolist() if col.startswith('sublex_')])+['origin']]
	df_suffixed = df_suffixed.rename(columns={col:format_sublex_name(col) for col in df_suffixed.columns.tolist() if col.startswith('sublex_')})
	df_suffixed.index = pd.CategoricalIndex(df_suffixed.index, df_suffixed.index, True)

	# df_suffixed.groupby('origin').plot.barh(stacked=True, subplots=True)
	num_origins = df_suffixed.origin.unique().size
	sizes_per_origin = df_suffixed.origin.value_counts(sort=False).tolist()
	# fig,axes = plt.subplots(nrows=num_origins, ncols=1, sharex=True, figsize=())
	fig = plt.figure(figsize=(9,9))
	gs = gridspec.GridSpec(num_origins,1,figure=fig,height_ratios=sizes_per_origin)
	axes = []
	# cm = LinearSegmentedColormap.from_list('reordered_tab10', ['C2','C5','C0'], N=3)
	# norm = BoundaryNorm(list(range(cm.N)), cm.N)
	if multinom_ps is None: 
		multinom_ps = [1.0/len(sublex_ids)]*len(sublex_ids) # Uniform null hypothesis
	if all_sublex:
		full_sublex_ids = ['sublex_{ix}'.format(ix=sublex_ix) for sublex_ix in range(len(multinom_ps))]
	else:
		full_sublex_ids = sorted(sublex_ids)
		multinom_ps = [multinom_ps[int(sublex_ix.split('_')[-1])] for sublex_ix in full_sublex_ids]
	multinom_ps_r = rpy2.robjects.FloatVector(multinom_ps)
	for g,(origin,sub_df) in zip(gs, df_suffixed.groupby('origin')):
		ax = plt.subplot(g)
		axes.append(ax)
		sub_df.plot.barh(stacked=True, ax=ax, legend=False)
		ax.invert_yaxis()
		ax.set_xlim((0,1))
		# cum_p = 0.0
		# for sublex_ix in sorted(sublex_ids):
		# 	cum_p += multinom_ps[int(sublex_ix.split('_')[-1])]
		# 	ax.axvline(x=cum_p, color = 'gray', linestyle='--')
		new_labels = []
		for y,ytl_txt in zip(ax.get_yticks(),ax.get_yticklabels()):
			suffix = ytl_txt.get_text()
			sub_df_originx = df_data[df_data.last_suffix==suffix]
			new_labels.append(suffix.replace('B','Adv').replace('_>',r'$\to$'))
			# N = sub_df_originx.shape[0]
			counts = sub_df_originx.map_sublex_base.value_counts().reindex(full_sublex_ids, fill_value=0).tolist()
			counts_r = rpy2.robjects.IntVector(counts)
			p_val = emt.multinomial_test(counts_r, multinom_ps_r)[-1][0] # p-value is the last item returned (so, -1), and "[0]" extracts the value as a Python float.
			# p_val = sstat.binom_test(num_2, n=N_2_and_5, p=multinom_ps, alternative='two-sided')
			if p_val < 0.001:
				significance = '***'
			elif p_val < 0.01:
				significance = '** '
			elif p_val < 0.05:
				significance = '*  '
			else:
				significance = ''# '   '
			# trans = ax.get_xaxis_transform()
			if significance:
				ax.annotate(significance, (1.01, y), annotation_clip=False, verticalalignment='center')
			# ax.annotate('{significance} (N={N_2_and_5: >4}/{N: >4}, p={p_val:.4f})'.format(N=N, N_2_and_5=N_2_and_5, p_val=p_val, significance=significance), (1.01, y), xycoords='data', annotation_clip=False)
		ax.set_title(origin+' suffixes')
		ax.set_yticklabels(new_labels)
		ax.set_title(origin+' suffixes')
	plt.subplots_adjust(hspace=0.5)
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right')
	ax.set_xlabel("Proportions of the bases' MAP categories")
	axes[num_origins//2].set_ylabel('Suffixes')
	fig.suptitle('MAP classification of the bases of the %i most common suffixes' % df_suffixed.shape[0])
	plt.savefig(os.path.join(result_dir,'bar_base-of-suffixed_MAP_derivational-corpus.png'), bbox_inches='tight')
	fig.clear()

	# df_suffixed.reset_index().rename(columns={'index':'suffix'}).to_csv(os.path.join(result_dir,'suffix_classification.tsv'), sep='\t', encoding='utf-8')
	# print df_suffixed.reset_index().rename(columns={'index':'suffix'})#.to_latex(index=False, float_format='{:,.2f}'.format)


def format_sublex_name(original):
	ix = int(original.split('_')[-1])
	ix2new_name = {0:'-ity',2:'Latinate',5:'Germanic'}
	if ix in ix2new_name:
		return r'\textsc{Sublex}\textsubscript{$\approx$' + ix2new_name[ix] + r'}'
	else:
		return original

	
	
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the data.')
	parser.add_argument('hdf_path', type=str, help='Path to the hdf5 file containing parameters of the trained mode. Used to obtain the p-value.')
	parser.add_argument('origin_path', type=str, help='Path to the csv file containing info about etymological origin of suffixes.')
	parser.add_argument('result_dir', type=str, help='Path to the directory where figure is saved.')
	parser.add_argument('-a', '--all_sublex', action='store_true', help='If selected, sublexica w/o MAP members are included in the computation of p-value.')
	args = parser.parse_args()
	df_data = pd.read_csv(args.data_path, encoding='utf-8')
	df_data = df_data[~df_data.map_sublex_base.isnull()]

	df_origin = pd.read_csv(args.origin_path)
	# df_origin['suffix'] = df_origin.suffix.map(lambda s: s if s.startswith('-') else '-' + s)

	df_stick = pd.read_hdf(args.hdf_path, key='/sublex/stick')
	log_assignment_probs = ppi.get_log_assignment_probs(df_stick)
	assignment_probs = np.exp(log_assignment_probs)

	bar(df_data, df_origin, args.result_dir, multinom_ps=assignment_probs, all_sublex=args.all_sublex)
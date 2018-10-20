# coding: utf-8

import numpy as np
import pandas as pd
import sys, os.path

def get_grouped_segments(df_atom, base_counts, threshold=1):
	df_atom = df_atom.sort_values('value')
	df_grouped_segments = pd.DataFrame(columns=['sublex_id','cluster_id','active_segments','num_active_segments'])
	for (sublex_id, cluster_id), df_atom_sub in df_atom.groupby(['sublex_id','cluster_id']):
		active_segments = df_atom_sub[(df_atom_sub.dirichlet_par - base_counts) >= threshold].decoded_value.tolist()
		num_active_segments = len(active_segments)
		df_grouped_segments = df_grouped_segments.append(
									pd.DataFrame(
										[[sublex_id, cluster_id, '_'.join(active_segments),num_active_segments]]
										,
										columns=['sublex_id','cluster_id','active_segments','num_active_segments']
									)
									,
									ignore_index=True
								)
	df_grouped_segments['non_singleton'] = df_grouped_segments.num_active_segments > 1
	return df_grouped_segments

def get_base_counts(log_path):
	target = False
	with open(log_path, 'r') as f:
		for line in f.readlines():
			line = line.rstrip()
			if 'Base count of top level Dirichlet: ' in line:
				target = True
				line = line.split('Base count of top level Dirichlet: ')[1]
				base_counts = []
			if target:
				base_counts += line.strip('[ ]').split('  ')
				if ']' in line:
					return map(
								np.float64,
								base_counts
							)

if __name__ == '__main__':
	result_dir = sys.argv[1]
	sublex_ids_str = sys.argv[2].split(',')
	sublex_ids = map(int, sublex_ids_str)

	base_counts = np.array(get_base_counts(os.path.join(result_dir, 'VI_DP_ngram.log')))
	
	df_atom = pd.read_hdf(os.path.join(result_dir, 'variational_parameters.h5'), key='/sublex/_1gram/context_/atom')
	df_atom = df_atom[df_atom.sublex_id.isin(sublex_ids)]
	
	df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	df_code.set_index('code', inplace=True)
	decoder = df_code.symbol.to_dict()

	df_atom['decoded_value'] = df_atom.value.map(decoder)

	df_grouped_segments = get_grouped_segments(df_atom, base_counts)
	df_grouped_segments = df_grouped_segments.sort_values(['num_active_segments','sublex_id'], ascending=[False,True])
	df_grouped_segments.to_csv(os.path.join(result_dir, 'grouped_segments_in-sublex-%s.csv' % '-'.join(sublex_ids_str)), index=False, encoding='utf-8')



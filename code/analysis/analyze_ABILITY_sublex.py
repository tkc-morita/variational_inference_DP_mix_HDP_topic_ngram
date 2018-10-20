# coding: utf-8

import pandas as pd
import sys
	
	
	
if __name__=='__main__':
	result_path=sys.argv[1]
	df_result = pd.read_csv(result_path)

	data_path = '../data/CelexLemmasInTranscription-DISC_modified.tsv'
	df_data = pd.read_csv(data_path, sep='\t')

	df = pd.concat([df_result, df_data], axis=1)

	target_sublex = 'sublex_0'
	suffix = '@bIl@tI' # -ability
	# suffix = '@tI' #-ty
	# suffix = '@bP' #-able

	is_in_target_sublex = df.most_probable_sublexicon == target_sublex
	ends_with_target_suffix = df.DISC.str.endswith(suffix)

	target_sublex_size = is_in_target_sublex.sum().astype(float)
	num_words_with_suffix = ends_with_target_suffix.sum().astype(float)

	num_both = (is_in_target_sublex & ends_with_target_suffix).sum().astype(float)

	print('sublex_size')
	print(target_sublex_size)
	print('# of words ending in {sf}'.format(sf=suffix))
	print(num_words_with_suffix)
	print('intersection')
	print(num_both)
	# target_sublex words all end in -ability? (precision)
	precision = num_both / target_sublex_size
	print('precision')
	print(precision)

	# All -ability words in target_sublex? (recall)
	recall = num_both / num_words_with_suffix
	print('recall')
	print(recall)

	exceptions = df[(~is_in_target_sublex) & (ends_with_target_suffix)]
	print(exceptions)
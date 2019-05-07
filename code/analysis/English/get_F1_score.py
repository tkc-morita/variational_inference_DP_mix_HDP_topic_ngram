# coding: utf-8

import pandas as pd
import argparse
import scipy.stats as sps
# import my_autopct



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the tsv file containing full data classified.')
	args = parser.parse_args()

	df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')

	print('Excluded:')
	print(df[df.most_probable_sublexicon.isnull()])

	print('Mixed:')
	print(df[df.MyEtym=='MIXED'])

	df = df[~df.most_probable_sublexicon.isnull()]
	df = df[df.MyEtym.isin(['Latin','non_Latin'])]
	print(df.MyEtym.value_counts())
	
	# linguistic_work = 'Levin1993_Latin'
	# linguistic_work = 'YangMontrul2017_Latin'
	# linguistic_work = 'YangMontrul2017_none'
	# my_etym = 'non_Latin'
	# df = df[df.linguistic_work == linguistic_work]
	# df = df[df.MyEtym == my_etym]

	grammatical = (df.double_object == 'grammatical')
	ungrammatical = (df.double_object == 'ungrammatical')

	num_grammatical = grammatical.astype(float).sum()
	num_ungrammatical = ungrammatical.astype(float).sum()


	print('===================Based on predicted sublexica==================')
	map_sublex_2 = (df.most_probable_sublexicon == 'sublex_2')
	map_sublex_5 = (df.most_probable_sublexicon == 'sublex_5')

	grammatical_sublex_5 = (grammatical & map_sublex_5)
	ungrammatical_sublex_2 = (ungrammatical & map_sublex_2)

	map_sublex_2_size = map_sublex_2.astype(float).sum()
	map_sublex_5_size = map_sublex_5.astype(float).sum()

	num_grammatical_sublex_5 = grammatical_sublex_5.astype(float).sum()
	num_ungrammatical_sublex_2 = ungrammatical_sublex_2.astype(float).sum()


	print('Grammatical as True Positive')
	precision = num_grammatical_sublex_5 / map_sublex_5_size
	print('precision', precision)
	recall = num_grammatical_sublex_5 / num_grammatical
	print('recall', recall)
	f = sps.hmean([precision, recall])
	print('f', f)

	print('Ungrammatical as True Positive')
	precision = num_ungrammatical_sublex_2 / map_sublex_2_size
	print('precision', precision)
	recall = num_ungrammatical_sublex_2 / num_ungrammatical
	print('recall', recall)
	f = sps.hmean([precision, recall])
	print('f', f)



	# Based on true sublexica
	print('===================Based on true sublexica==================')

	latin = (df.MyEtym == 'Latin')
	non_latin = (df.MyEtym == 'non_Latin')

	grammatical_non_latin = (grammatical & non_latin)
	ungrammatical_latin = (ungrammatical & latin)

	latin_size = latin.astype(float).sum()
	non_latin_size = non_latin.astype(float).sum()

	num_grammatical_non_latin = grammatical_non_latin.astype(float).sum()
	num_ungrammatical_latin = ungrammatical_latin.astype(float).sum()


	print('Grammatical as True Positive')
	precision = num_grammatical_non_latin / non_latin_size
	print('precision', precision)
	recall = num_grammatical_non_latin / num_grammatical
	print('recall', recall)
	f = sps.hmean([precision, recall])
	print('f', f)

	print('Ungrammatical as True Positive')
	precision = num_ungrammatical_latin / latin_size
	print('precision', precision)
	recall = num_ungrammatical_latin / num_ungrammatical
	print('recall', recall)
	f = sps.hmean([precision, recall])
	print('f', f)
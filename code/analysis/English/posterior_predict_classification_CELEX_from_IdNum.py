# coding: utf-8

import pandas as pd
import os.path, argparse


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('assignment_path', type=str, help='Path to the csv file containing classification info.')
	parser.add_argument('data_path', type=str, help='Path to the tsv file containing full data classified.')
	parser.add_argument('--subdata', type=str, default=None, help='Path to a file containing subdata to be output.')
	parser.add_argument('--sep', type=str, default='\t', help='Separator of the subdata file. tab stop by default (i.e., tsv).')
	args = parser.parse_args()


	df_assign = pd.read_csv(args.assignment_path)

	df_data = pd.read_csv(args.data_path, encoding='utf-8', sep='\t')

	# df_assign.loc[:,'lemma'] = df_data.lemma
	df_assign.loc[:,'DISC'] = df_data.DISC
	df_assign.loc[:,'IdNum'] = df_data.loc[:,'rank']

	if args.subdata is None:
		datafile_name = os.path.splitext(os.path.basename(args.data_path))[0]
	else:
		df_subdata = pd.read_csv(args.subdata, encoding='utf-8', sep=args.sep)
		df_assign = pd.merge(df_subdata, df_assign, how='left', on='IdNum')
		df_assign = df_assign.drop(columns='customer_id').drop_duplicates()
		datafile_name = os.path.splitext(os.path.basename(args.subdata))[0]

	result_dir = os.path.dirname(args.assignment_path)
	classification_filename = datafile_name+'_posterior-classification.tsv'
	df_assign.to_csv(os.path.join(result_dir, classification_filename), index=False, encoding = 'utf-8', sep='\t')

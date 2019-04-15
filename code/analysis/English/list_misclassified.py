# coding: utf-8

import pandas as pd
import argparse, os.path


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
	df_result['origin']=df_data.origin
	df_result['DISC'] = df_data.DISC
	# df_result['DISC_csv'] = df_data.DISC_csv
	df_result['lemma'] = df_data.lemma

	df_misclassified_Germanic = df_result[df_result.origin.isin(['AngloSaxon', 'OldNorse', 'Dutch']) & (df_result.most_probable_sublexicon=='sublex_2')]
	df_misclassified_Germanic = df_misclassified_Germanic.sort_values('sublex_2', ascending=False)
	df_misclassified_Latinate = df_result[df_result.origin.isin(['Latin', 'French']) & (df_result.most_probable_sublexicon=='sublex_5')]
	df_misclassified_Latinate = df_misclassified_Latinate.sort_values('sublex_5', ascending=False)
	df_misclassified = pd.concat([df_misclassified_Germanic, df_misclassified_Latinate], axis=0)
	df_misclassified.to_csv(os.path.join(result_dir, 'misclassified.csv') , sep=',', encoding='utf-8', index=False)

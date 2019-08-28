# coding: utf-8

import pandas as pd
import argparse, os.path

nasals = [u'nː',u'mː',u'9ː',u'ŋː',u'ɲː']
voiceless = [u't',u's',u'ç',u'h',u'ʦ',u'ʨ',u'ɕ',u'k',u'ɸ',u'p']

def include_NC(listed_symbols):
	for s1,s2 in zip(listed_symbols, listed_symbols[1:]):
		if s1 in nasals and s2 in voiceless:
			return True
	return False

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the data.')
	parser.add_argument('class_path', type=str, help='Path to the classification results.')
	parser.add_argument('save_path', type=str, help='Path to the csv file where results are saved.')
	args = parser.parse_args()

	df_data = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')
	df_class = pd.read_csv(args.class_path)
	df_data.loc[:,'most_probable_sublexicon'] = df_class.most_probable_sublexicon

	df_data = df_data[(df_data.wType==u'和') & df_data.IPA_csv.map(lambda s: include_NC(s.split(',')))]

	print(df_data.most_probable_sublexicon.value_counts())

	df_data.to_csv(args.save_path, index=False, encoding='utf-8')
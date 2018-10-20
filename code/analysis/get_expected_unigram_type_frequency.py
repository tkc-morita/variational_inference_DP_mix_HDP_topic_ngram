# coding: utf-8


import pandas as pd
import sys, os.path


if __name__ == '__main__':
	bigram_csv_path = sys.argv[1]
	df = pd.read_csv(bigram_csv_path, encoding='utf-8')

	df['type_frequency'] = (df['expected_frequency'] >= 1).astype(float)

	df_unigram = df.groupby(by=['value','sublex_id']).type_frequency.sum()

	df_unigram.to_csv(os.path.join(os.path.split(bigram_csv_path)[0], 'unigram_type_freq.csv'), encoding='utf-8')
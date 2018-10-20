# coding: utf-8

import pandas as pd
import itertools, sys

def count_segments(data_csv):
	words=[word.split(',') for index, word in data_csv.iteritems()]
	return len(list(itertools.chain.from_iterable(words)))


if __name__ == '__main__':
	data_path = sys.argv[1]
	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	print df.shape[0]
	print count_segments(df.IPA_csv)
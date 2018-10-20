# coding: utf-8

import pandas as pd
import sys


def count_substring(word, substring):
	substring_as_list = substring.split(',')
	substring_len = len(substring_as_list)
	word_as_list = word.split(',')
	word_len = len(word_as_list)
	count = 0
	for ix in range(word_len):
		if word_as_list[ix:ix+substring_len] == substring_as_list:
			count += 1
	return count


if __name__ == '__main__':
	df = pd.read_csv(sys.argv[1], encoding='utf-8', sep='\t')

	substrings = [u'9ː,i', u'ɯ,w', u'i,w', u'ä,w', u'ʦ', u'ʦːː']
	

	for sbs in substrings:
		print(sbs).encode('utf-8')
		print('total')
		print(df.IPA_csv.map(lambda x: count_substring(x, sbs)).sum())
		print('\n')
		for wType, sub_df in df.groupby('wType'):
			print(sbs).encode('utf-8')
			print(wType).encode('utf-8')
			print(sub_df.IPA_csv.map(lambda x: count_substring(x, sbs)).sum())
			print('\n')

	print('e,END')
	print('total')
	print(df.IPA_csv.str.endswith('e').sum())
	print('\n')
	for wType, sub_df in df.groupby('wType'):
		print('e,END')
		print(wType).encode('utf-8')
		print(sub_df.IPA_csv.str.endswith('e').sum())
		print('\n')
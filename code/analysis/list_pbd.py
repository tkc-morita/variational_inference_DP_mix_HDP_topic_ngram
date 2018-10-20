# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path


if __name__ == '__main__':
	result_path = sys.argv[1]
	result_dir, filename = os.path.split(result_path)
	n = int(filename.split('gram')[0][-1])

	# sublex_id = int(sys.argv[2])

	# list_length = int(sys.argv[3])
	
	df_ngram = pd.read_csv(result_path)

	

	min_frequency = 1
	# df_ngram = df_ngram[df_ngram.sublex_id == sublex_id]
	df_ngram = df_ngram[df_ngram.frequency >= min_frequency]

	# nasals = ['nː','mː','9ː','ŋː','ɲː']
	# voiceless = ['t','s','ç','h','ʦ','ʨ','ɕ','k','ɸ','p']
	ps = ['pːː','p','bːː','b','dːː','d']
	df_ngram = df_ngram[df_ngram.value.isin(ps)]
	# df_ngram = df_ngram.sort_values('representativeness', ascending=False)
	# df_ngram = df_ngram.tail(n=list_length)
	df_ngram = df_ngram.sort_values(
					[
						# 'context',
						'value',
						'sublex_id'
						]
						)

	# df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	# df_code.set_index('code', inplace=True)
	# decoder = df_code.symbol.to_dict()

	# df_ngram['decoded_value'] = df_ngram.value.map(decoder)
	# df_ngram['decoded_context'] = df_ngram.context.map(lambda context: '_'.join(map(lambda code: decoder[int(code)], context.split('_'))))


	df_ngram.to_csv(os.path.join(result_dir, 'pbd_freq-at-least-%s.csv' % (str(min_frequency))), index=False, encoding='utf-8')



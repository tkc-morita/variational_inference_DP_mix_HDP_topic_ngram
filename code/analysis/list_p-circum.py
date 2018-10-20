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
	df_ngram = pd.read_csv(result_path, encoding='utf-8')
	df_ngram = df_ngram[df_ngram.frequency >= min_frequency]
	# df_ngram.loc[:,'decoded_value'] = df_ngram.decoded_value.str.replace(u'ä','a')
	df_context = df_ngram.decoded_context.str.split('_', expand=True).rename(columns={0:'c1', 1:'c2'})
	df_ngram = pd.concat([df_ngram,df_context], axis=1)

	df_ngram['density'] = df_ngram.expected_frequency / df_ngram.frequency


	df_ngram = df_ngram[(df_ngram.c2==u'p')]
	df_ngram.sort_values(
		[
			'density',
			'sublex_id'
		]
		,
		inplace=True,
		ascending=False
	)


	df_ngram.to_csv(os.path.join(result_dir, 'p-circum_freq-at-least-%s.csv' % (str(min_frequency))), index=False, encoding='utf-8')



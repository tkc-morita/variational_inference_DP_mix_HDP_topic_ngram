# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path


def convert_context(context):
	if context.startswith('__'):
		context = context.replace('__', '_|')
	elif context.endswith('__'):
		context = context.replace('__', '|_')
	else:
		context = context.replace('_', '|')
	
	return context

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
	df_ngram = df_ngram.sort_values(['context', 'value', 'sublex_id'])

	df_ngram['context'] = df_ngram.decoded_context # .map(convert_context)
	df_ngram['value'] = df_ngram.decoded_value

	value = [
		'i',
		'!',
		'#',
		"$",
		"3",
		"1",
		"2",
		"4",
		"5",
		"6",
		"7",
		"8",
		"9",
		"K",
		"M",
		"I",
		"Y",
		"E",
		"{",
		"&",
		"A",
		"Q",
		"V",
		"O",
		"U",
		"@",
		"c",
		"q",
		"0",
		"~",
		"u",
			]
	df_ngram_init_V = df_ngram[
						(
							(df_ngram.context.str.startswith('START_START'))
							&
							(df_ngram.value.isin(value))
						)
						|
						(
							(df_ngram.context.str.startswith('START'))
							&
							(~df_ngram.context.map(lambda x: x.split('_')[0] in value))
							&
							(df_ngram.value.isin(value))
						)
						].groupby(['value','sublex_id']).expected_frequency.sum()
	df_ngram_init_V = df_ngram_init_V.reset_index()
	# df_ngram = df_ngram.sort_values('representativeness', ascending=False)
	# df_ngram = df_ngram.tail(n=list_length)

	# df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	# df_code.set_index('code', inplace=True)
	# decoder = df_code.symbol.to_dict()

	# df_ngram['decoded_value'] = df_ngram.value.map(decoder)
	# df_ngram['decoded_context'] = df_ngram.context.map(lambda context: '_'.join(map(lambda code: decoder[int(code)], context.split('_'))))

	
	df_ngram_init_V.to_csv(os.path.join(result_dir, 'START-X-V_freq-at-least-%s.csv' % (str(min_frequency))), index=False, encoding='utf-8')
	# df_ngram_init_CV.to_csv(os.path.join(result_dir, 'START-C-V_freq-at-least-%s.csv' % (str(min_frequency))), index=False, encoding='utf-8')



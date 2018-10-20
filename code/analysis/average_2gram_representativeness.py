# coding: utf-8

import pandas as pd
import numpy as np
import sys, os.path

def average_over_context(df_ngram, context_length):
	columns = ['sublex_id','context','value','representativeness','expected_frequency','frequency']
	new_df = pd.DataFrame(columns=columns)
	for (sublex_id, context, value), sub_df in df_ngram.groupby(['sublex_id','context_%i' % (context_length-1), 'decoded_value']):
		mean_representativeness = sub_df.representativeness.mean()
		expected_frequency = sub_df.expected_frequency.sum()
		frequency = sub_df.frequency.sum()
		new_df = new_df.append(
							pd.DataFrame(
								[[
									sublex_id,
									context,
									value,
									mean_representativeness,
									expected_frequency,
									frequency
								]]
								,
								columns=columns
							)
							,
							ignore_index=True
						)
	return new_df



def split_ngram_context(df_ngram):
	df_split_context = df_ngram.decoded_context.str.split('_', expand=True)
	context_length = df_split_context.shape[1]
	df_split_context = df_split_context.rename(columns={i:'context_%i' % i for i in range(context_length)})
	df_ngram = pd.concat(
				[df_ngram,
				df_split_context],
				axis=1
				)
	return df_ngram, context_length


if __name__ == '__main__':
	path = sys.argv[1]
	result_dir = os.path.split(path)[0]
	df_ngram = pd.read_csv(path, encoding='utf-8')

	df_ngram, context_length = split_ngram_context(df_ngram)
	df_ngram_shorter = average_over_context(df_ngram, context_length)

	df_ngram_shorter.to_csv(os.path.join(result_dir, 'averaged-2gram-representativeness.csv'), encoding='utf-8', index=False)


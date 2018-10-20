# coding: utf-8

import pandas as pd
import numpy as np
import sys, os.path


def update_cumulative_probs(df_ngram, df_cumulative, context_length):
	df_merged = pd.merge(
					df_ngram,
					df_cumulative,
					how='left',
					on=['context_%i' % i for i in range(context_length)]+['sublex_id']
					)
	# print 'cum', df_cumulative[df_cumulative['context_1']==3].cum_prob
	# print 'merged', df_merged[df_merged['context_1']==3].cum_prob
	df_merged['cum_prob'] = df_merged.cum_prob * df_merged.prob
	return df_merged.groupby(
			['context_%i' % i for i in range(1,context_length)]
			+
			['value','sublex_id']
			).cum_prob.sum().reset_index().rename(
				columns={('context_%i' % i):('context_%i' % (i-1)) for i in range(1,context_length)}
				).rename(columns={'value':'context_%i' % (context_length-1)})


def main_loop(df_ngram, max_length):
	df_ngram, context_length = get_ngram_split_context(df_ngram)
	df_cumulative = df_ngram.groupby(
						['context_%i' % i for i in range(context_length)]
							+
							['sublex_id','context']
							).prob.sum().reset_index().rename(
								columns={'prob':'cum_prob'}
							)
	df_cumulative['cum_prob'] = np.float64(0)
	start_code = df_ngram.value.max()+1
	df_cumulative.loc[df_cumulative['context_%i' % (context_length-1)] == start_code, 'cum_prob'] = np.float64(1)
	df_prob_lengths = pd.DataFrame(columns=['cum_prob','sublex_id','length'])
	for length in xrange(max_length):
		df_cumulative = update_cumulative_probs(df_ngram, df_cumulative, context_length)
		sub_df_prob_length = df_cumulative[df_cumulative['context_%i' % (context_length-1)] == 0].groupby('sublex_id').cum_prob.sum().reset_index()
		sub_df_prob_length['length'] = length
		df_prob_lengths = df_prob_lengths.append(sub_df_prob_length, ignore_index=True)
	return df_prob_lengths





def get_ngram_split_context(df_ngram):
	df_split_context = df_ngram.context.str.split('_', expand=True).astype(int)
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
	df_ngram = pd.read_csv(path)

	max_length = int(sys.argv[2])

	df_length = main_loop(df_ngram, max_length)
	df_length.to_csv(os.path.join(result_dir, 'posterior_length_probs.csv'), index=False)


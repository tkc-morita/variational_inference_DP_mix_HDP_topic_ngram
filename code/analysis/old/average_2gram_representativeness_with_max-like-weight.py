# coding: utf-8

import pandas as pd
import numpy as np
import sys, os.path

def average_over_context(df_ngram, context_length, sublex_assignment_probs):
	sublex_assignment2others_probs = 1.0 - sublex_assignment_probs
	columns = ['sublex_id','context','value','expected_frequency','frequency','prob','prob_x_assignment','prob_x_assignment2others']
	new_df = pd.DataFrame(columns=columns)
	for (sublex_id, context), df_per_context in df_ngram.groupby(['sublex_id','context_%i' % (context_length-1)]):
		prob_denominator = df_per_context.expected_frequency.sum()
		for value, df_per_2gram in df_per_context.groupby('decoded_value'):
			expected_frequency = df_per_2gram.expected_frequency.sum()
			frequency = df_per_2gram.frequency.sum()
			if prob_denominator == 0:
				prob = np.float64(0)
			else:
				prob = expected_frequency / prob_denominator
			prob_x_assignment = prob * sublex_assignment_probs[sublex_id]
			prob_x_assignment2others = prob * sublex_assignment2others_probs[sublex_id]
			new_df = new_df.append(
								pd.DataFrame(
									[[
										sublex_id,
										context,
										value,
										expected_frequency,
										frequency,
										prob,
										prob_x_assignment,
										prob_x_assignment2others
									]]
									,
									columns=columns
								)
								,
								ignore_index=True
							)
	for (context, value), sub_df in new_df.groupby(['context', 'value']):
		for sublex_id, df_per_sublex in sub_df.groupby('sublex_id'):
			new_df.loc[
				(new_df.context==context)
				&
				(new_df.value==value)
				&
				(new_df.sublex_id==sublex_id)
				,
				'representativeness'
				] = np.log(df_per_sublex.prob_x_assignment2others) - np.log(np.sum(sub_df[sub_df.sublex_id!=sublex_id].prob_x_assignment))
	return new_df.loc[:,['sublex_id','context','value','expected_frequency','frequency','prob','representativeness']]



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


def get_assignment_probs(df_stick):
	df_stick = df_stick.sort_values('cluster_id')
	df_stick['beta_sum'] = df_stick.beta_par1 + df_stick.beta_par2
	df_stick['stop_prob'] = df_stick.beta_par1 / df_stick.beta_sum
	df_stick['pass_prob'] = df_stick.beta_par2 / df_stick.beta_sum
	assignment_probs = []
	cum_pass_prob = 1.0
	for row_tuple in df_stick.itertuples():
		assignment_probs.append(row_tuple.stop_prob * cum_pass_prob)
		cum_pass_prob *= row_tuple.pass_prob
	assignment_probs.append(cum_pass_prob)
	assignment_probs = np.array(assignment_probs)

	return assignment_probs

if __name__ == '__main__':
	path = sys.argv[1]
	result_dir = os.path.split(path)[0]
	df_ngram = pd.read_csv(path, encoding='utf-8')

	df_ngram, context_length = split_ngram_context(df_ngram)

	hdf5_path = os.path.join(result_dir, 'variational_parameters.h5')
	df_stick = pd.read_hdf(hdf5_path, key='/sublex/stick')
	assignment_probs = get_assignment_probs(df_stick)


	df_ngram_shorter = average_over_context(df_ngram, context_length, assignment_probs)

	df_ngram_shorter.to_csv(os.path.join(result_dir, 'averaged-2gram-representativeness_max-like.csv'), encoding='utf-8', index=False)


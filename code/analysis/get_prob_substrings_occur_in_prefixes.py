# coding: utf-8

"""
p(a | Z=t) (unigram) and p(a2 | a1, Z=t) (bigram) are hard to compute exactly given trigram probabilities.
This program approximates them by marginalizing over all the prefixes up to some freely parameterized length l.
"""

import numpy as np
import pandas as pd
import itertools, argparse

def get_prob_substring_occur_in_prefixes(ngram_array, max_length):
	"""
	Get the probability of substrings a_1,...,a_n appearing (at least once) in the first max_length symbols given sublex.
	Probs. of longer substrings a_1,...,a_n,a_{n+1},... can be computed later by further multiplying trigram probs.
	"""
	n = ngram_array.ndim - 1
	assert max_length > n - 1, 'Too short max_length={max_length}'.format(max_length=max_length)

	start_code = ngram_array.shape[-2] - 1
	
	# Initialize the probs.
	prob_after_k_trans = np.zeros_like(ngram_array)
	sub_array = ngram_array
	sub_cum = prob_after_k_trans
	for context_pos in range(n - 1):
		sub_array = sub_array[start_code]
		sub_cum = sub_cum[start_code]
	sub_cum[:] = sub_array

	prob_absent = 1.0 - prob_after_k_trans

	for k in range(2,max_length+1):
		marg_over_oldest_trans = np.sum(prob_after_k_trans,axis=0)
		prob_after_k_trans = (
						marg_over_oldest_trans.reshape(
							marg_over_oldest_trans.shape[:-1]
							+ (1,marg_over_oldest_trans.shape[-1])
						) # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array # shape=(vocab_size)^n x num_sublex
						)
		prob_absent *= 1.0 - prob_after_k_trans
	return 1.0 - prob_absent # i.e. prob. of existance.


def get_ngram_trans_array(df_ngram):
	n = len(df_ngram.context.ix[0].split('_')) + 1
	vocab_size = df_ngram.value.drop_duplicates().size + 1 # # of phonemes + END + START.
	num_sublexica = df_ngram.sublex_id.drop_duplicates().size
	ngram_array = np.zeros([vocab_size]*n + [num_sublexica])
	for context in itertools.chain(
						itertools.product(range(1,vocab_size-1), repeat=n-1),
						get_init_contexts(n, vocab_size)
						):
		sub_array = ngram_array
		for c in context:
			sub_array = sub_array[c]
		sub_array[:-1,:] = df_ngram[
							df_ngram.context == '_'.join(map(str, context))
							].sort_values(
								['value','sublex_id']
							).prob.values.reshape(vocab_size-1,num_sublexica)
	return ngram_array

def get_init_contexts(n, vocab_size):
	"""
	START is assumed be indexed by the largest integer.
	"""
	contexts = []
	start_code = vocab_size-1
	for init_length in range(1,n):
		init_context = (start_code,) * init_length
		rest_length = n - 1 - init_length
		contexts += [init_context+rest for rest in itertools.product(range(1,start_code), repeat=rest_length)]
	return contexts


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('ngram_path', type=str, help='Path to the csv file containing ngram prob. info.')
	parser.add_argument('max_length', type=int, help='Maximum length of the prefixes to (n-1)-gram substrings whose marginalization would approximate the prob. of the (n-1)-ngrams.')
	parser.add_argument('save_path', type=str, help='Path to the csv file to save results.')
	args = parser.parse_args()

	df_ngram = pd.read_csv(args.ngram_path, encoding='utf-8')
	
	ngram_array = get_ngram_trans_array(df_ngram)

	joint_probs_of_n_minus_1_grams = get_prob_substring_occur_in_prefixes(ngram_array, args.max_length)

	n = joint_probs_of_n_minus_1_grams.ndim
	index = pd.MultiIndex.from_product(
				[range(s) for s in joint_probs_of_n_minus_1_grams.shape],
				names=['symbol_{pos}'.format(pos=pos) for pos in range(n-1)]+['sublex']
				)
	df_result = pd.DataFrame(joint_probs_of_n_minus_1_grams.reshape(-1,1), columns=['prob'], index=index).reset_index()
	df_result.to_csv(args.save_path, index=False, encoding='utf-8')
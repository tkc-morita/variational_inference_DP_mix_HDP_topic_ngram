# coding: utf-8

"""
p(a | Z=t) (unigram) and p(a2 | a1, Z=t) (bigram) are hard to compute exactly given trigram probabilities.
This program approximates them by marginalizing over all the prefixes up to some freely parameterized length l.
"""

import numpy as np
import pandas as pd
import argparse, sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import get_prob_substrings_occur_in_prefixes, encode_decode, get_substring_representativeness, parse_vi



def mask_VX_transitions(ngram_array, coded_vowels):
	for v in coded_vowels:
		ngram_array[...,v:,:,:] = 0.0
	return ngram_array

VOWELS_IN_DISC = list('IE{VQU@i#$u312456789cq0~')
# CONSONANTS_IN_DISC = list('pbtdkgNmnlrfvTDszSZjxhwJ_CFHPR')

def mask_last_C(ngram_array, coded_consonants):
	for c in coded_consonants:
		ngram_array[...,c,:] = 0.0
	return ngram_array

def get_prob_init_V_given_polysyllabicity(ngram_array, max_length_init_cc, max_length_internal_cc, coded_vowels, coded_consonants):
	"""
	Get the probability of word-initial vowel (V1) given sublex and presence of another vowel (V2) in the word.
	V1 is maximally preceded by max_length_init_cc consonants, and V1 and V2 are intervened by 0 to max_length_internal_cc consonants.
	"""
	n = ngram_array.ndim - 1

	start_code = ngram_array.shape[-2] - 1
	
	# Initialize the probs.
	prob_after_k_trans = np.zeros_like(ngram_array)
	sub_array = ngram_array
	sub_cum = prob_after_k_trans
	for context_pos in range(n - 1):
		sub_array = sub_array[start_code]
		sub_cum = sub_cum[start_code]
	sub_cum[:] = sub_array # p(x | start*(n-1))

	prob_absent = 1.0 - prob_after_k_trans

	ngram_array_VX_masked = mask_VX_transitions(ngram_array, coded_vowels)

	prob_init_V_given_polysyllabicity = hoge(mask_last_C(prob_after_k_trans.copy()).sum(axis=0), ngram_array, max_length_internal_cc, ngram_array_VX_masked=ngram_array_VX_masked, coded_vowels=coded_vowels)

	for k in range(max_length_init_cc+1):
		marg_over_oldest_trans = np.sum(prob_after_k_trans,axis=0)
		prob_after_k_trans = (
						marg_over_oldest_trans.reshape(
							marg_over_oldest_trans.shape[:-1]
							+ (1,marg_over_oldest_trans.shape[-1])
						) # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array_VX_masked # shape=(vocab_size)^n x num_sublex
						)
		prob_init_V_given_polysyllabicity += hoge(mask_last_C(prob_after_k_trans.copy()).sum(axis=0), ngram_array, max_length_internal_cc, ngram_array_VX_masked=ngram_array_VX_masked, coded_vowels=coded_vowels)
	return prob_init_V_given_polysyllabicity # i.e. prob. of existance.

def hoge(init_array, ngram_array, max_length_cc, ngram_array_VX_masked=None, coded_vowels=None):
	if ngram_array_VX_masked is None:
		if coded_vowels is None:
			raise ValueError('Either ngram_array_VX_masked or coded_vowels needs to be specified.')
		else:
			ngram_array_VX_masked = mask_VX_transitions(ngram_array, coded_vowels)
	prob_after_k_trans = (
						init_array.reshape(
							init_array.shape[:-1]
							+ (1,init_array.shape[-1])
						) # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array # shape=(vocab_size)^n x num_sublex
						)
	prefix_length = init_array.ndim-2
	if max_length_cc < prefix_length:
		for k in range(1,max_length_cc):
			prob_after_k_trans = (
						marg_over_oldest_trans.reshape(
							marg_over_oldest_trans.shape[:-1]
							+ (1,marg_over_oldest_trans.shape[-1])
						) # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array_VX_masked # shape=(vocab_size)^n x num_sublex
						)
			marg_over_oldest_trans = np.sum(prob_after_k_trans,axis=0)
		prob_after_k_trans = (
								marg_over_oldest_trans.reshape(
									marg_over_oldest_trans.shape[:-1]
									+ (1,marg_over_oldest_trans.shape[-1])
								) # shape=(vocab_size)^(n-1) x 1 x num_sublex
								*
								ngram_array # shape=(vocab_size)^n x num_sublex
								)
		# marg_over_prefix = prob_after_k_trans.sum(axis=tuple(range(prefix_length - max_length_cc)))
	else:
		for k in range(1, prefix_length):
			prob_after_k_trans = (
							marg_over_oldest_trans.reshape(
								marg_over_oldest_trans.shape[:-1]
								+ (1,marg_over_oldest_trans.shape[-1])
							) # shape=(vocab_size)^(n-1) x 1 x num_sublex
							*
							ngram_array_VX_masked # shape=(vocab_size)^n x num_sublex
							)
			marg_over_oldest_trans = np.sum(prob_after_k_trans,axis=0)
		for k in range(prefix_length, max_length_cc):
			prob_after_k_trans = (
							prob_after_k_trans.reshape(
								prob_after_k_trans.shape[:-1]
								+ (1,prob_after_k_trans.shape[-1])
							) # shape=(vocab_size)^n x 1 x num_sublex
							*
							ngram_array_VX_masked.reshape((1,)+ngram_array_VX_masked.shape)# shape= 1 x (vocab_size)^n x num_sublex
							)
			marg_over_oldest_trans = np.sum(prob_after_k_trans,axis=1) # Skip the target axis.
		prob_after_k_trans = (
							prob_after_k_trans.reshape(
								prob_after_k_trans.shape[:-1]
								+ (1,prob_after_k_trans.shape[-1])
							) # shape=(vocab_size)^n x 1 x num_sublex
							*
							ngram_array.reshape((1,)+ngram_array.shape)# shape= 1 x (vocab_size)^n x num_sublex
							)
		return prob_after_k_trans.sum(axis=tuple(range(1,prob_after_k_trans.ndim-1)))
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('ngram_path', type=str, help='Path to the csv file containing ngram prob. info.')
	parser.add_argument('code_path', type=str, help='Path to the csv file containing segment coding.')
	parser.add_argument('max_CC_length', type=int, help='Maximum length of the initial consonant cluster before the first vowel.')
	parser.add_argument('save_path', type=str, help='Path to the csv file to save results.')
	args = parser.parse_args()

	df_ngram = pd.read_csv(args.ngram_path, encoding='utf-8')
	
	ngram_array = get_prob_substrings_occur_in_prefixes.get_ngram_trans_array(df_ngram)
	
	df_code = pd.read_csv(args.code_path, encoding='utf-8')
	encoder, decoder = encode_decode.df2coder(df_code)
	coded_vowels = [encoder[v] for v in VOWELS_IN_DISC]
	# coded_consonants = [encoder[c] for c in CONSONANTS_IN_DISC]

	ngram_array = mask_VX_transitions(ngram_array, coded_vowels)


	joint_probs_of_n_grams = get_prob_substrings_occur_in_prefixes.get_prob_substring_occur_in_prefixes(ngram_array, args.max_CC_length+1)

	
	
	probs_last_symbol = np.sum(
							joint_probs_of_n_grams.reshape(
								-1,
								joint_probs_of_n_grams.shape[-2],
								joint_probs_of_n_grams.shape[-1]
								),
							axis=0
							)

	log_assignment_probs = parse_vi.get_log_sublex_assignment_probs(os.path.dirname(args.code_path))
	log_assignment_over_others = get_substring_representativeness.get_log_assignment_over_others(log_assignment_probs)


	results = []
	for v in VOWELS_IN_DISC:
		probs_v = probs_last_symbol[encoder[v],:]
		representativeness = get_substring_representativeness.get_representativeness(np.log(probs_v), log_assignment_over_others)
		results += [(v,sublex_ix,prob,rep) for sublex_ix,(prob,rep) in enumerate(zip(probs_v,representativeness))]
	df_result = pd.DataFrame(results, columns=['vowel','sublex','prob','representativeness'])
	df_result.to_csv(args.save_path, index=False, encoding='utf-8')
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
	mask = np.zeros_like(ngram_array, dtype=int)
	# for v in coded_vowels:
	mask[...,coded_vowels,:,:] = 1
	return np.ma.masked_array(ngram_array, mask=mask)

VOWELS_IN_DISC = list('IE{VQU@i#$u312456789cq0~')
# CONSONANTS_IN_DISC = list('pbtdkgNmnlrfvTDszSZjxhwJ_CFHPR')

def mask_last_non_V(ngram_array, coded_vowels):
	mask = np.ones_like(ngram_array, dtype=int)
	# for c in coded_vowels:
	mask[...,coded_vowels,:] = 0
	return np.ma.masked_array(ngram_array, mask=mask)

def get_prob_init_V_given_polysyllabicity(ngram_array, max_length_init_cc, max_length_internal_cc, coded_vowels):
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

	ngram_array_VX_masked = mask_VX_transitions(ngram_array, coded_vowels)

	prob_init_V_given_polysyllabicity = get_prob_V_before_CstarV(mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=0), ngram_array, max_length_internal_cc, coded_vowels, ngram_array_VX_masked=ngram_array_VX_masked)

	for k in range(1, max_length_init_cc+1):
		prob_after_k_trans = (
						np.sum(prob_after_k_trans,axis=0)[...,np.newaxis,:] # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array_VX_masked # shape=(vocab_size)^n x num_sublex
						)
		prob_init_V_given_polysyllabicity += get_prob_V_before_CstarV(mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=0), ngram_array, max_length_internal_cc, coded_vowels, ngram_array_VX_masked=ngram_array_VX_masked)
	return prob_init_V_given_polysyllabicity.filled(fill_value=0.0)

def get_prob_V_before_CstarV(init_context_w_last_c_masked, ngram_array, max_length_cc, coded_vowels, ngram_array_VX_masked=None):
	if ngram_array_VX_masked is None:
		ngram_array_VX_masked = mask_VX_transitions(ngram_array, coded_vowels)
	# The only V-to-X transision.
	prob_after_k_trans = (
						init_context_w_last_c_masked[...,np.newaxis,:] # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array # shape=(vocab_size)^n x num_sublex
						)
	prefix_length = init_context_w_last_c_masked.ndim-2
	prob_init_V_given_polysyllabicity = mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=tuple(range(prefix_length))+(-2,))
	prefix_reminder = prefix_length
	if max_length_cc < prefix_length:
		for k in range(1,max_length_cc+1):
			prob_after_k_trans = (
						np.sum(prob_after_k_trans,axis=0)[...,np.newaxis,:] # shape=(vocab_size)^(n-1) x 1 x num_sublex
						*
						ngram_array_VX_masked # shape=(vocab_size)^n x num_sublex
						)
			prefix_reminder -= 1
			prob_init_V_given_polysyllabicity += mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=
										tuple(range(prefix_reminder))
										+
										tuple(range(-k-2,-1)) # -2: second V, -1: sublex
										)
	else:
		for k in range(1, prefix_length):
			prob_after_k_trans = (
							np.sum(prob_after_k_trans,axis=0)[...,np.newaxis,:] # shape=(vocab_size)^(n-1) x 1 x num_sublex
							*
							ngram_array_VX_masked # shape=(vocab_size)^n x num_sublex
							)
			prefix_reminder -= 1
			prob_init_V_given_polysyllabicity += mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=
										tuple(range(prefix_reminder))
										+
										tuple(range(-k-2,-1)) # -2: second V, -1: sublex
										)
		prob_after_k_trans = (
							np.sum(prob_after_k_trans,axis=0)[...,np.newaxis,:] # shape=(vocab_size)^(n-1) x 1 x num_sublex
							*
							ngram_array_VX_masked # shape= 1 x (vocab_size)^n x num_sublex
							) # The 0th axis is the first Vs now.
		prob_init_V_given_polysyllabicity += mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=tuple(range(1,prob_after_k_trans.ndim-1)))
		for k in range(prefix_length+1, max_length_cc+1):
			prob_after_k_trans = (
							prob_after_k_trans[...,np.newaxis,:] # shape=(vocab_size)^n x 1 x num_sublex
							*
							ngram_array_VX_masked.reshape((1,)+ngram_array_VX_masked.shape)# shape= 1 x (vocab_size)^n x num_sublex
							).sum(axis=1)
			prob_init_V_given_polysyllabicity += mask_last_non_V(prob_after_k_trans, coded_vowels).sum(axis=tuple(range(1,prob_after_k_trans.ndim-1)))
	return prob_init_V_given_polysyllabicity

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('ngram_path', type=str, help='Path to the csv file containing ngram prob. info.')
	parser.add_argument('code_path', type=str, help='Path to the csv file containing segment coding.')
	parser.add_argument('max_length_init_cc', type=int, help='Maximum length of the initial consonant cluster before the first vowel.')
	parser.add_argument('max_length_internal_cc', type=int, help='Maximum length of the consonant cluster between the first and second vowels.')
	parser.add_argument('save_path', type=str, help='Path to the csv file to save results.')
	args = parser.parse_args()

	df_ngram = pd.read_csv(args.ngram_path, encoding='utf-8')
	
	ngram_array = get_prob_substrings_occur_in_prefixes.get_ngram_trans_array(df_ngram)
	
	df_code = pd.read_csv(args.code_path, encoding='utf-8')
	encoder, decoder = encode_decode.df2coder(df_code)
	coded_vowels = [encoder[v] for v in VOWELS_IN_DISC]


	prob_init_V_given_polysyllabicity = get_prob_init_V_given_polysyllabicity(ngram_array, args.max_length_init_cc, args.max_length_internal_cc, coded_vowels)


	log_assignment_probs = parse_vi.get_log_sublex_assignment_probs(os.path.dirname(args.code_path))
	log_assignment_over_others = get_substring_representativeness.get_log_assignment_over_others(log_assignment_probs)


	results = []
	for v in VOWELS_IN_DISC:
		probs_v = prob_init_V_given_polysyllabicity[encoder[v],:]
		representativeness = get_substring_representativeness.get_representativeness(np.log(probs_v), log_assignment_over_others)
		results += [(v,sublex_ix,prob,rep) for sublex_ix,(prob,rep) in enumerate(zip(probs_v,representativeness))]
	df_result = pd.DataFrame(results, columns=['vowel','sublex','prob','representativeness'])
	df_result.to_csv(args.save_path, index=False, encoding='utf-8')
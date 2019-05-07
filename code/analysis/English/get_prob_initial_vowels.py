# coding: utf-8

"""
p(a | Z=t) (unigram) and p(a2 | a1, Z=t) (bigram) are hard to compute exactly given trigram probabilities.
This program approximates them by marginalizing over all the prefixes up to some freely parameterized length l.
"""

import numpy as np
import pandas as pd
import itertools, argparse, sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import get_prob_substrings_occur_in_prefixes, encode_decode, get_substring_representativeness, parse_vi



def mask_VC_transitions(ngram_array, coded_vowels, coded_consonants):
	for v,c in itertools.product(coded_vowels, coded_consonants):
		ngram_array[...,v,c] = 0.0
	return ngram_array

VOWELS_IN_DISC = list('IE{VQU@i#$u312456789cqO~')
CONSONANTS_IN_DISC = list('pbtdkgNmnlrfvTDszSZjxhwJ_CFHPR')

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
	coded_consonants = [decoder[c] for c in CONSONANTS_IN_DISC]

	ngram_array = mask_VC_transitions(ngram_array, coded_vowels, coded_consonants)


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
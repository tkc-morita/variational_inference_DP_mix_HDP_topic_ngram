# coding: utf-8

import numpy as np
import scipy.misc as spm
import itertools



def posterior_predict_classification(data, df_ngram, log_assignment_probs, n, start_code):
	classification_list = []
	df_ngram = df_ngram.sort_values('sublex_id')
	for word in data:
		log_joint_obs_and_assignment = get_log_word_segments_probs(
												word,
												df_ngram,
												n,
												start_code
												) + log_assignment_probs
		classification_list.append(
									np.exp(
										log_joint_obs_and_assignment
										-
										spm.logsumexp(log_joint_obs_and_assignment)
									)
								)
	return np.array(classification_list)


def posterior_predict_symbol_in_word(
						prefixes,
						targets,
						controls,
						suffixes,
						df_ngram,
						log_assignment_probs,
						n,
						start_code
						):
	log_ratio_list = []
	df_ngram = df_ngram.sort_values('sublex_id')
	for prefix,target,control,suffix in zip(prefixes, targets, controls, suffixes):
		log_marginal_like_target = spm.logsumexp(
											get_log_word_segments_probs(
														prefix+target+suffix,
														df_ngram,
														n,
														start_code
													)
											+
											log_assignment_probs
											)
		log_marginal_like_control = spm.logsumexp(
											get_log_word_segments_probs(
														prefix+control+suffix,
														df_ngram,
														n,
														start_code
													)
											+
											log_assignment_probs
											)
		
		log_ratio_list.append(
								log_marginal_like_target - log_marginal_like_control
								)
	return np.array(log_ratio_list)

def posterior_predict_symbol_in_word_by_prefix_MAP(
						prefixes,
						targets,
						controls,
						suffixes,
						df_ngram,
						log_assignment_probs,
						n,
						start_code
						):
	log_ratio_list = []
	df_ngram = df_ngram.sort_values('sublex_id')
	prefix_map_sublexica = np.argmax(
								posterior_predict_classification(prefixes, df_ngram, log_assignment_probs, n, start_code)
								,
								axis=1
								)
	for prefix,target,control,suffix,sublex_id in zip(prefixes, targets, controls, suffixes, prefix_map_sublexica):
		log_map_like_target = (
											get_log_word_segments_probs(
														target+suffix,
														df_ngram,
														n,
														start_code,
														additional_context=prefix
													)
											+
											log_assignment_probs
											)[sublex_id]
		log_map_like_control = (
											get_log_word_segments_probs(
														control+suffix,
														df_ngram,
														n,
														start_code,
														additional_context=prefix
													)
											+
											log_assignment_probs
											)[sublex_id]
		
		log_ratio_list.append(
								log_map_like_target - log_map_like_control
								)
	return np.array(log_ratio_list)

def get_log_posterior_predict_prob_of_target_and_control(
						prefixes,
						targets,
						controls,
						suffixes,
						df_ngram,
						log_assignment_probs,
						n,
						start_code,
						inventory
						):
	log_prob_list = []
	df_ngram = df_ngram.sort_values('sublex_id')
	for prefix,target,control,suffix in zip(prefixes, targets, controls, suffixes):
		log_prefix_like = get_log_word_segments_probs(
														prefix,
														df_ngram,
														n,
														start_code
													)

		log_marginal_like_target = spm.logsumexp(
											log_prefix_like
											+
											get_log_word_segments_probs(
														target+suffix,
														df_ngram,
														n,
														start_code,
														additional_context=prefix
													)
											+
											log_assignment_probs
											)
		log_marginal_like_control = spm.logsumexp(
											log_prefix_like
											+
											get_log_word_segments_probs(
														control+suffix,
														df_ngram,
														n,
														start_code,
														additional_context=prefix
													)
											+
											log_assignment_probs
											)
		log_normalizer = spm.logsumexp(
							log_prefix_like
							+
							spm.logsumexp(
								[get_log_word_segments_probs(
											segments+suffix,
											df_ngram,
											n,
											start_code,
											additional_context=prefix
										)
								for segments
								in itertools.product(inventory, repeat=len(target))
								]
								,
								axis=0
								)
							+
							log_assignment_probs
							)
		log_pred_prob_target = log_marginal_like_target - log_normalizer
		log_pred_prob_control = log_marginal_like_control - log_normalizer
		log_prob_list.append([log_pred_prob_target,log_pred_prob_control])
	return np.array(log_prob_list)

def get_unnormalized_log_posterior_predict_prob_of_target(
						words,
						df_ngram,
						log_assignment_probs,
						n,
						start_code
						):
	log_prob_list = []
	df_ngram = df_ngram.sort_values('sublex_id')
	for word in words:
		log_marginal_like_target = spm.logsumexp(
										get_log_word_segments_probs(
														word,
														df_ngram,
														n,
														start_code
													)
										+
										log_assignment_probs
										)
		log_prob_list.append(log_marginal_like_target)
	return np.array(log_prob_list)

def get_log_posterior_predict_prob_of_target(
						prefixes,
						targets,
						suffixes,
						df_ngram,
						log_assignment_probs,
						n,
						start_code,
						inventory
						):
	log_prob_list = []
	df_ngram = df_ngram.sort_values('sublex_id')
	for prefix,target,suffix in zip(prefixes, targets, suffixes):
		log_prefix_like = get_log_word_segments_probs(
														prefix,
														df_ngram,
														n,
														start_code
													)
		log_marginal_like_target = spm.logsumexp(
											log_prefix_like
											+
											get_log_word_segments_probs(
														target+suffix,
														df_ngram,
														n,
														start_code,
														additional_context=prefix
													)
											+
											log_assignment_probs
											)
		log_normalizer = spm.logsumexp(
							log_prefix_like
							+
							spm.logsumexp(
								[get_log_word_segments_probs(
											segments+suffix,
											df_ngram,
											n,
											start_code,
											additional_context=prefix
										)
								for segments
								in itertools.product(inventory, repeat=len(target))
								]
								,
								axis=0
								)
							+
							log_assignment_probs
							)
		log_pred_prob_target = log_marginal_like_target - log_normalizer
		log_prob_list.append(log_pred_prob_target)
	return np.array(log_prob_list)

def get_log_word_segments_probs(word, df_ngram, n, start_code, additional_context=()):
	init_context = (start_code,)*(n-1)
	init_context += additional_context
	return np.sum(
			[
			get_log_ngram_probs(ngram, df_ngram)
			for ngram
			in zip(*[
					(init_context[-(n-1):]+word)[i:] for i in range(n)
					])
			]
			,
			axis=0
			)

def get_log_ngram_probs(ngram, df_ngram):
	context = '_'.join(map(str,ngram[:-1]))
	value = ngram[-1]
	sub_df_ngram = df_ngram[
						(df_ngram.context == context)
						&
						(df_ngram.value == value)
						]
	return np.log(sub_df_ngram.prob.values)

def get_log_assignment_probs(df_stick):
	df_stick = df_stick.sort_values('cluster_id')
	df_stick.loc[:,'beta_sum'] = df_stick.beta_par1 + df_stick.beta_par2
	df_stick.loc[:,'log_stop_prob'] = np.log(df_stick.beta_par1) - np.log(df_stick.beta_sum)
	df_stick.loc[:,'log_pass_prob'] = np.log(df_stick.beta_par2) - np.log(df_stick.beta_sum)
	log_assignment_probs = []
	log_cum_pass_prob = 0
	for row_tuple in df_stick.itertuples():
		log_assignment_probs.append(row_tuple.log_stop_prob + log_cum_pass_prob)
		log_cum_pass_prob += row_tuple.log_pass_prob
	log_assignment_probs.append(log_cum_pass_prob)
	log_assignment_probs = np.array(log_assignment_probs)
	return log_assignment_probs

# def encode_data(data, encoder):
# 	return [tuple([encoder[symbol] for symbol in string.split(',')+['END']]) for string in data]



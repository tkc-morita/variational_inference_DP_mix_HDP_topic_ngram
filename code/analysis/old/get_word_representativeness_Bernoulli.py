# coding: utf-8

import pandas as pd
import numpy as np
import scipy.misc as spm
import scipy.special as sps
# import scipy.stats as spst
import sys, os.path

def main_loop(encoded_data, log_ngram, n, start_code, log_assignment_over_others):
	return np.array(
			[get_representativeness(
						string,
						log_ngram,
						n,
						start_code,
						log_assignment_over_others,
						)
			for string
			in encoded_data
			]
			)

def get_representativeness(
			string,
			log_ngram,
			n,
			start_code,
			log_assignment_over_others,
			):
	log_likes = (
					get_log_like(string, log_ngram, n, start_code)
				)
	return log_likes - spm.logsumexp(log_likes[np.newaxis,:] + log_assignment_over_others, axis=1)


def get_log_like(string, log_ngram, n, start_code):
	return np.sum([
				log_ngram['_'.join(map(str, ngram_window[:-1]))][ngram_window[-1]]
				for ngram_window in zip(*[((start_code,)*(n-1)+string)[i:] for i in range(n)])
			]
			,
			axis=0
			)

def encode_data(data, encoder):
	return [tuple([encoder[symbol] for symbol in string.split(',')+['END']]) for string in data]

def get_log_assignment_over_others(df_stick):
	df_stick = df_stick.sort_values('cluster_id')
	df_stick['beta_sum'] = df_stick.beta_par1 + df_stick.beta_par2
	df_stick['log_stop_prob'] = np.log(df_stick.beta_par1) - np.log(df_stick.beta_sum)
	df_stick['log_pass_prob'] = np.log(df_stick.beta_par2) - np.log(df_stick.beta_sum)
	log_assignment_probs = []
	log_cum_pass_prob = 0
	for row_tuple in df_stick.itertuples():
		log_assignment_probs.append(row_tuple.log_stop_prob + log_cum_pass_prob)
		log_cum_pass_prob += row_tuple.log_pass_prob
	log_assignment_probs.append(log_cum_pass_prob)
	log_assignment_probs = np.array(log_assignment_probs)


	num_sublex = log_assignment_probs.size
	log_assignment_to_others = np.repeat(log_assignment_probs[np.newaxis,:], num_sublex, axis=0)
	np.fill_diagonal(log_assignment_to_others, -np.inf)
	log_assignment_to_others = spm.logsumexp(log_assignment_to_others, axis=1)
	log_assignment_over_others = log_assignment_probs[np.newaxis,:] - log_assignment_to_others[:,np.newaxis]
	np.fill_diagonal(log_assignment_over_others, -np.inf)

	return log_assignment_over_others



# class GammaPoisson(object):
# 	def __init__(self, df):
# 		df = df.sort_values('sublex_id')
# 		self.num_failures = df['shape'].values
# 		p = 1 / (df.rate.values+np.float64(1))
		
# 		self.log_p = np.log(p)
# 		self.gammaln_num_failure = sps.gammaln(self.num_failures)
# 		self.num_failures_x_log_1_minus_p = self.num_failures * np.log(1-p)


# 	def get_log_prob(self, num_success):
# 		return (
# 			sps.gammaln(num_success+self.num_failures)
# 			-
# 			sps.gammaln(num_success+1)
# 			-
# 			self.gammaln_num_failure
# 			+
# 			num_success * self.log_p
# 			+
# 			self.num_failures_x_log_1_minus_p
# 		)


def get_log_ngram_probs(df_ngram):
	df_ngram = df_ngram.sort_values('sublex_id')
	log_ngram = {}
	for (context,value), sub_df in df_ngram.groupby(['context','value']):
		if context in log_ngram.keys():
			log_ngram[context][value] = np.log(sub_df.prob.values)
		else:
			log_ngram[context]={value: np.log(sub_df.prob.values)}
	start_code = df_ngram.value.max()+1
	return log_ngram, start_code


if __name__ == '__main__':
	ngram_path = sys.argv[1]
	data_path = sys.argv[2]

	result_dir,filename = os.path.split(ngram_path)
	n = int(filename.split('gram')[0].split('_')[-1])

	df_ngram = pd.read_csv(ngram_path)
	log_ngram, start_code = get_log_ngram_probs(df_ngram)

	df_data = pd.read_csv(data_path, sep='\t', encoding='utf-8')

	df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	df_code.set_index('symbol', inplace=True)
	encoder = df_code.code.to_dict()

	encoded_data = encode_data(df_data.IPA_csv.tolist(), encoder)

	hdf5_path = os.path.join(result_dir, 'variational_parameters.h5')
	df_stick = pd.read_hdf(hdf5_path, key='/sublex/stick')
	log_assignment_over_others = get_log_assignment_over_others(df_stick)


	representativeness = main_loop(encoded_data, log_ngram, n, start_code, log_assignment_over_others)
	df_rep = pd.DataFrame(
					representativeness,
					columns=['sublex_%i' % i for i in range(representativeness.shape[1])]
					)
	df_rep['word_id'] = df_rep.index
	df_rep.to_csv(
						os.path.join(result_dir, 'word_representativeness.csv')
						,
						index=False
					)





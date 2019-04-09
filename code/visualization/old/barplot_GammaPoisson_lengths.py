# coding: utf-8

import numpy as np
import scipy.stats as spst
import scipy.special as sps
import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path

def get_nbinom_parameters(df):
	df['num_failures'] = df['shape']
	df['p'] = 1 / (df.rate+np.float64(1))
	# df['log_p'] = -np.log(df.rate+np.float64(1))

def barplot_negative_binomial(df, result_dir, num_bins):
	x = np.arange(num_bins)
	for sublex_tuple in df.itertuples(index=False):
		plt.bar(x, np.exp(get_log_negative_binomial_prob(x, sublex_tuple.num_failures, sublex_tuple.p)), 1)
		# print sublex_tuple.sublex_id, sublex_tuple.shape / sublex_tuple.rate
		# print sublex_tuple.p
		# print spst.nbinom.pmf(x, sublex_tuple.num_failures, sublex_tuple.p)
		# print np.exp(get_log_negative_binomial_prob(x, sublex_tuple.num_failures, sublex_tuple.p))
		plt.title('Gamma-Poisson posterior predictive probability mass in sublex %i' % sublex_tuple.sublex_id)
		plt.xlabel('(Segmental) word Lengths')
		plt.ylabel('Posterior predictive probability')
		plt.savefig(os.path.join(result_dir, 'Gamma-Poisson-lengths_sublex-%i.png' % sublex_tuple.sublex_id))
		plt.gcf().clear()

def barplot_GammaPoisson_lengths(df, result_dir, num_bins):
	get_nbinom_parameters(df)
	barplot_negative_binomial(df, result_dir, num_bins)

def get_log_negative_binomial_prob(num_success, num_failures, p):
	return (
			sps.gammaln(num_success+num_failures)
			-
			sps.gammaln(num_success+1)
			-
			sps.gammaln(num_failures)
			+
			num_success * np.log(p)
			+
			num_failures * np.log(1-p)
		)



if __name__ == '__main__':
	result_dir = sys.argv[1]
	num_bins = int(sys.argv[2])

	df = pd.read_hdf(os.path.join(result_dir, 'variational_parameters.h5'), key='/sublex/length')

	barplot_GammaPoisson_lengths(df, result_dir, num_bins)


	
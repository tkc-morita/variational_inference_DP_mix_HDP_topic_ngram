# coding: utf-8

import numpy as np
import scipy.misc as spm
import itertools



class Poisson_Binomial(object):
	"""
	Poisson_Binomial distribution.
	"""
	def __init__(self, ps):
		self.ps = np.array(ps)
		self.one_minus_ps = 1 - self.ps
		self.log_ps = np.log(self.ps)
		self.log_one_minus_ps = np.log(self.one_minus_ps)
		self.n = self.ps.size

	def _get_prob_vector(self):
		return [
			
			for k in range(0, self.n)
		]

	def _get_prob(self, k):
		


	def pmf(self,k):
		range_n = range(self.n)
		set_n = set(range_n)
		return np.sum(
			[
				np.prod(
					np.take(self.ps, k_indeces)
				)
				*
				np.prod(
					np.take(self.one_minus_ps, list(set_n - set(k_indeces)))
				)
				#
				for k_indeces in itertools.combinations(xrange(self.n), k)
				for i in k_indeces
				for j in (set_n - set(k_indeces))
			]
		)

	def log_pmf(self, k):
		range_n = range(self.n)
		set_n = set(range_n)
		return spm.logsumexp(
			[
				np.sum(
					np.take(self.log_ps, k_indeces)
				)
				*
				np.sum(
					np.take(self.log_one_minus_ps, list(set_n - set(k_indeces)))
				)
				#
				for k_indeces in itertools.combinations(xrange(self.n), k)
				for i in k_indeces
				for j in (set_n - set(k_indeces))
			]
		)

	def mean(self):
		return np.sum(self.ps)

	def variance(self):
		return np.sum(self.ps * self.one_minus_ps)


	def sample(self):
		weights = np.array([self.pmf(k) for k in xrange(self.n + 1)])
		return np.random.choice(np.arange(self.n + 1), p=weights)
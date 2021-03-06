# coding: utf-8

import numpy as np
import scipy.special as sps
import scipy.misc as spm
import itertools,sys,datetime,os
from logging import getLogger,FileHandler,DEBUG,Formatter
import pandas as pd
# import warnings



logger = getLogger(__name__)

def update_log_handler(log_path):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	handler = FileHandler(filename=os.path.join(log_path,'VI_DP_ngram.log'))	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	logger.info("Logger (re)set up.")
	


class VariationalInference(object):
	def __init__(
			self,
			num_sublex,
			segmental_data,
			accentual_data,
			segmental_data_missing_accent,
			n,
			T_base,
			concent_priors,
			dirichlet_concentration,
			result_path,
			inventory_size=None # without START or END
			):
		update_log_handler(result_path)
		
		logger.info('DP mixture of words.')
		self.n=n
		logger.info('The base distribution is INDEPENDENT %i-gram with DP backoff with Poisson length.' % n)
		logger.info('Log files prior to 2017/10/20 have incorrectly state the base distribution is SHARED.')
		logger.info('Results prior to 2017/10/20 have wrong columns on Poisson variational parameters: shape and rate are flipped.')
		
		logger.info('Long vowels and geminates are now (from 03/16/2017) treated as independent segments.')
		logger.info('Script last updated at %s'
							% datetime.datetime.fromtimestamp(
									os.stat(sys.argv[0]).st_mtime
									).strftime('%Y-%m-%d-%H:%M:%S')
							)
		
		global num_symbols
		if inventory_size is None:
			num_symbols = len(set(reduce(lambda x,y: x+y, segmental_data + segmental_data_missing_accent))) + 1
		else:
			num_symbols = inventory_size + 1
		logger.info('# of symbols: %i' % num_symbols)

		dirichlet_base_counts = dirichlet_concentration * np.ones(num_symbols)
		
		
		self.segmental_data = [Word(word, n, word_id) for word_id,word in enumerate(segmental_data)]
		self.segmental_data_missing_accent = [Word(word, n, word_id, missing=True) for word_id,word in enumerate(segmental_data_missing_accent)]
		data_size = len(segmental_data)
		data_size_missing_acccent = len(segmental_data_missing_accent)
		logger.info('# of words with accent info: %i' % data_size)
		logger.info('# of words W/O accent info: %i' % data_size_missing_acccent)
		logger.info('# of words in total: %i' % (data_size + data_size_missing_acccent))

		
		self.varpar_assignment = np.random.dirichlet(np.ones(num_sublex), data_size) # phi in Blei and Jordan.
		self.varpar_assignment_missing = np.random.dirichlet(np.ones(num_sublex), data_size_missing_acccent)
		
		
		self.varpar_concent =VarParConcent(concent_priors)

		self.varpar_stick = np.random.gamma(1,#self.concent_priors[0],
												20,#self.concent_priors[1],
												size=(num_sublex-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, np.newaxis]

		self.varpar_concent.add_dp(self)

		self.num_clusters = num_sublex
		logger.info('(max) # of tables for sublexicalization: %i' % num_sublex)
		
		full_contexts = set([ngram.context
								for word in self.segmental_data+self.segmental_data_missing_accent
								for ngram in word.ngrams
								])

		self.hdp_ngram = HDPNgram(
							T_base,
							n,
							concent_priors,
							dirichlet_base_counts,
							full_contexts,
							num_sublex,
							self.segmental_data,
							self.segmental_data_missing_accent,
							self
							)

		self.beta_bernoulli = Beta_Bernoulli(
								(
									1.0,
									1.0
								),
								accentual_data,
								data_size_missing_acccent,
								num_sublex,
								self
							)
		
		self.hdp_ngram.set_varpar_assignment()
		self._update_word_likelihood()
		
		logger.info('# of tables: %i' % T_base)
		logger.info('Gamma priors on concent_priorsration: (%f,%f)'
							% (concent_priors[0],concent_priors[1]**-1)
							)
		logger.info('Base count of top level Dirichlet: %s' % str(dirichlet_base_counts))

		self.result_path = result_path 
		logger.info('Initialization complete.')


		
	
		
	
	def train(self, max_iters, min_increase):
		logger.info("Main loop started.")
		logger.info("Max iteration is %i." % max_iters)
		logger.info("Will be terminated if variational bound is only improved by <=%f." % min_increase)
		converged=False
		iter_id=0
		self.current_var_bound = self.get_var_bound()
		while iter_id<max_iters:
			iter_id+=1
			self.update_varpars()
			logger.info("Variational parameters updated (Iteration ID: %i)." % iter_id)
			new_var_bound = self.get_var_bound()
			improvement = new_var_bound-self.current_var_bound
			logger.info("Current var_bound is %0.12f (%+0.12f)." % (new_var_bound,improvement))
			if np.isnan(new_var_bound):
				raise Exception("nan detected.")
			if improvement<0:
				logger.error("variational bound decreased. Something wrong.")
				raise Exception("variational bound decreased. Something wrong.")
			elif improvement<=min_increase:
				converged = True
				break
			else:
				self.current_var_bound=new_var_bound
		if converged:
			logger.info('Converged after %i iterations.' % iter_id)
		else:
			logger.error('Failed to converge after max iterations.')
		logger.info('Final variational bound is %f.' % self.current_var_bound)
		
	def get_log_posterior_pred(self, test_data):
		pass
		
	def save_results(self, decoder):

		inventory=[('symbol_code_%i' % code) for code in decoder.keys()]#[symbol.encode('utf-8') for symbol in decoder.values()]
		# Shared HDP ngram
		with pd.HDFStore(
				os.path.join(self.result_path,'variational_parameters.h5')
				) as hdf5_store:
			df_concent = pd.DataFrame(columns = ['shape', 'rate', 'DP_name'])
			for context_length,level in self.hdp_ngram.tree.iteritems():
				vpc = self.hdp_ngram.varpar_concents[context_length]
				df_concent_sub = pd.DataFrame(
									vpc.rate[:,np.newaxis]
									,
									columns=['rate']
									)
				df_concent_sub['shape'] = vpc.shape
				df_concent_sub['DP_name'] = [('%igram_%i' % (context_length+1,sublex_id))
												for sublex_id
												in xrange(self.num_clusters)
												]
				df_concent = df_concent.append(df_concent_sub, ignore_index=True)
				for context,rst in level.iteritems():
					coded_context = '_'.join(map(str,context))#.encode('utf-8')

					if context_length != self.n-1:
						children_contexts = pd.Series(['_'.join([str(code) for code in child_dp.context])
													for child_dp in rst.children])
						df_assignment = pd.DataFrame(
										rst.varpar_assignment.flatten()[:,np.newaxis]
										,
										columns=["p"]
										)
						df_assignment['children_DP_context']=children_contexts.iloc[
																np.repeat(
																	np.arange(rst.varpar_assignment.shape[0])
																	,
																	np.prod(
																		rst.varpar_assignment.shape[1:]
																		)
																	)
																].reset_index(drop=True)
						df_assignment['sublex_id'] = np.tile(
														np.repeat(
															np.arange(rst.varpar_assignment.shape[1])
															,
															np.prod(rst.varpar_assignment.shape[2:])
															)
														,
														rst.varpar_assignment.shape[0]
														)
						df_assignment['children_cluster_id']=np.tile(
																np.repeat(
																	np.arange(rst.varpar_assignment.shape[2])
																	,
																	rst.varpar_assignment.shape[3]
																	)
																,
																np.prod(rst.varpar_assignment.shape[:2])
																)
						df_assignment['cluster_id']=np.tile(
														np.arange(rst.varpar_assignment.shape[3])
														,
														np.prod(rst.varpar_assignment.shape[:-1])
													)
						hdf5_store.put(
							("sublex/_%igram/context_%s/assignment"
							% (context_length+1,coded_context))
							,
							df_assignment
							,
	# 						encoding="utf-8"
							)
			
					num_sublex = rst.varpar_stick.shape[0]
					num_clusters = rst.varpar_stick.shape[1]
					df_stick=pd.DataFrame(rst.varpar_stick.reshape(
												num_sublex*num_clusters
												,
												2
											),
											columns=('beta_par1','beta_par2')
											)
					df_stick['sublex_id'] = np.repeat(np.arange(num_sublex), num_clusters)
					df_stick['cluster_id'] = np.tile(np.arange(num_clusters), num_sublex)
					hdf5_store.put(
						("sublex/_%igram/context_%s/stick"
						% (context_length+1,coded_context))
						,
						df_stick
						,
# 						encoding="utf-8"
						)
			
					
				

			df_atom = pd.DataFrame(
						self.hdp_ngram.tree[0][()].varpar_atom.flatten()[:,np.newaxis],
						columns=['dirichlet_par']
						)
			num_sublex,num_clusters,num_symbols = self.hdp_ngram.tree[0][()].varpar_atom.shape
			df_atom['sublex_id']=np.repeat(
										np.arange(num_sublex)
										,
										num_clusters*num_symbols
									)
			df_atom['cluster_id'] = np.tile(
											np.repeat(
												np.arange(num_clusters),
												num_symbols
											)
											,
											num_sublex
										)
			df_atom['value']=pd.Series(
								np.tile(
									np.arange(num_symbols)
									,
									num_sublex*num_clusters
								)
								)
			
			hdf5_store.put(
							'sublex/_1gram/context_/atom'
							,
							df_atom
							)



			# Sublex
			df_assignment_sl = pd.DataFrame(
									self.varpar_assignment
									,
									columns=[("sublex_%i" % table_id)
												for table_id in range(self.num_clusters)
												]
									)
			df_assignment_sl['most_probable_sublexicon']=df_assignment_sl.idxmax(axis=1)
			df_assignment_sl['customer_id']=df_assignment_sl.index
			df_assignment_sl.to_csv(os.path.join(self.result_path, "SubLexica_assignment_for-data-with-accent.csv"), index=False, encoding='utf-8')
			hdf5_store.put(
							"sublex/assignment/with_accent_info",
							df_assignment_sl,
# 							encoding='utf-8'
							)
			df_assignment_sl = pd.DataFrame(
									self.varpar_assignment_missing
									,
									columns=[("sublex_%i" % table_id)
												for table_id in range(self.num_clusters)
												]
									)
			df_assignment_sl['most_probable_sublexicon']=df_assignment_sl.idxmax(axis=1)
			df_assignment_sl['customer_id']=df_assignment_sl.index
			df_assignment_sl.to_csv(os.path.join(self.result_path, "SubLexica_assignment_for-data-WO-accent.csv"), index=False, encoding='utf-8')
			hdf5_store.put(
							"sublex/assignment/WO_accent_info",
							df_assignment_sl,
# 							encoding='utf-8'
							)
		
			df_stick_sl = pd.DataFrame(self.varpar_stick, columns=('beta_par1','beta_par2'))
			df_stick_sl['cluster_id']=df_stick_sl.index
			hdf5_store.put(
							"sublex/stick",
							df_stick_sl,
							)
		
			df_concent = df_concent.append(
									pd.DataFrame(
										[[
											self.varpar_concent.rate,
											self.varpar_concent.shape,
											'word_sublexicalization'
										]]
										,
										columns=['rate', 'shape', 'DP_name']
									)
								)

			hdf5_store.put(
							"sublex/concentration",
							df_concent,
# 							encoding='utf-8'
							)

			df_accent_rate = pd.DataFrame(
								self.beta_bernoulli.varpar_rate
								,
								columns=['unaccented','accented']
								)
			df_accent_rate['sublex_id'] = df_accent_rate.index
			hdf5_store.put(
							"sublex/accent/rate",
							df_accent_rate,
							)

			df_accent_missing = pd.DataFrame(
											self.beta_bernoulli.varpar_categorical
												,
												columns=['unaccented','accented']
											)
			df_accent_missing['customer_id'] = df_accent_missing.index
			hdf5_store.put(
							"sublex/accent/prediction_for_missing",
							df_accent_missing,
							)



		pd.DataFrame(decoder.items(),
						columns=('code','symbol')
						).to_csv(
							os.path.join(
								self.result_path,
								"symbol_coding.csv"
								),
							encoding='utf-8'
							,
							index=False
							)
		
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment[...,:-1], axis=0)+1
		self.varpar_stick[...,1] = np.cumsum(
										np.sum(
											self.varpar_assignment[:,:0:-1],
											axis=0
											)
										)[::-1]+self.varpar_concent.mean
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, np.newaxis]
		

	def _update_varpar_assignment(self):
		prior_terms = (
						np.append(
							self.E_log_stick[:,0],
							0
							)
						+
						np.append(
							0,
							np.cumsum(
								self.E_log_stick[:,1]
								)
							)
						)[np.newaxis,:]
		self.varpar_assignment = (
									prior_terms
									+
									self.word_likelihood
									)
		self.varpar_assignment=np.exp(self.varpar_assignment-spm.logsumexp(self.varpar_assignment, axis=-1)[:,np.newaxis])

		self.varpar_assignment_missing = (
											prior_terms
											+
											self.word_likelihood_missing
										)
		self.varpar_assignment_missing = np.exp(self.varpar_assignment_missing - spm.logsumexp(self.varpar_assignment_missing, axis=-1)[:,np.newaxis])

	def _update_word_likelihood(self):
		self.word_likelihood = (
									np.array(
										[
											word.get_E_log_likelihoods() # Output an array of length num_sublex
												for word in self.segmental_data
										]
										)
									+
									self.beta_bernoulli.get_log_like()
									)
		self.word_likelihood_missing = (
									np.array(
										[
											word.get_E_log_likelihoods() # Output an array of length num_sublex
												for word in self.segmental_data_missing_accent
										]
										)
									+
									self.beta_bernoulli.get_log_like_missing()
									)

	def update_varpars(self):
		self.varpar_concent.update()
		self._update_varpar_stick()
		self.hdp_ngram.update_varpars()
		self.beta_bernoulli.update_varpars()
		self._update_word_likelihood()
		self._update_varpar_assignment()
		
		
	def get_var_bound(self):
		"""
		Calculate the KL divergence bound based on the current variational parameters.
		We ignore the constant terms.
		"""
		return (
				self.varpar_concent.get_var_bound()
				+
				self.hdp_ngram.get_var_bound()
				+
				self.beta_bernoulli.get_var_bound()
				+
				self.get_sum_E_log_p_varpars()
				-
				self.get_E_log_q_varpar()
				)

	def get_sum_E_log_p_varpars(self):
		return (
				(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,1]) # E[alpha-1]*E[log (1-V)]
				+
				np.sum(
					self.E_log_stick[:,1]*np.cumsum(
												np.sum(
													self.varpar_assignment[...,:0:-1]
													,
													axis=0
													)
												+
												np.sum(
													self.varpar_assignment_missing[...,:0:-1]
													,
													axis=0
													)
												)[::-1]
					+
					self.E_log_stick[:,0]*(
						np.sum(self.varpar_assignment[:,:-1], axis=0)
						+
						np.sum(self.varpar_assignment_missing[:,:-1], axis=0)
						)
				) # E[log p(Z | V)]
				+
				np.sum(
					self.word_likelihood # num_words x num_sublex
					*
					self.varpar_assignment
					) # E[log p(X | Z,eta)]
				+
				np.sum(
					self.word_likelihood_missing
					*
					self.varpar_assignment_missing
				)
				) # E[log p(V, alpha)]


	def get_E_log_q_varpar(self):
		return (
					# E[log q(V)] below
					np.sum(self.E_log_stick[:,0]*(self.varpar_stick[:,0]-1))
					+
					np.sum(self.E_log_stick[:,1]*(self.varpar_stick[:,1]-1))
					-
					np.sum(
						sps.gammaln(self.varpar_stick),
						)
					+
					np.sum(sps.gammaln(self.sum_stick)) 
					+
					np.sum(
						self.varpar_assignment*np.ma.log(self.varpar_assignment)
						) # E[log q(Z)]
					+
					np.sum(
						self.varpar_assignment_missing*np.ma.log(self.varpar_assignment_missing)
						)
				)
		



		
class HDPNgram(object):
	def __init__(self, num_clusters, n, concent_priors, dirichlet_base_counts, full_contexts, num_sublex, data, data_missing, wrapper):
		self.wrapper=wrapper
		self.n = n
		self.tree = {k:{} for k in xrange(n)}
		self.varpar_concents = [VarParConcent_sublex(concent_priors, num_sublex) for context_length in xrange(n)]
		
		self.tree[0][()]=DP_top(
							num_clusters,
							0,
							(),
							self.varpar_concents[0],
							dirichlet_base_counts,
							num_sublex,
							self
							)
		

		for context_length, vpc in enumerate(self.varpar_concents[1:-1], start=1):
			for context in set(fc[n-1-context_length:] for fc in full_contexts):
				self.tree[context_length][context]\
								= DP(
									num_clusters,
									self.tree[context_length-1][context[1:]],
									context_length,
									context,
									vpc,
									num_sublex,
									self
									)

		for context in full_contexts:
			self.tree[self.n-1][context]\
				= DP_bottom(
					num_clusters,
					self.tree[n-2][context[1:]],
					n-1,
					context,
					self.varpar_concents[-1],
					num_sublex,
					self
					)
		
		[ngram.enter_a_restaurant(self.tree[self.n-1][ngram.context])
			for word in data
			for ngram in word.ngrams
			]
		[ngram.enter_a_restaurant(self.tree[self.n-1][ngram.context])
			for word in data_missing
			for ngram in word.ngrams
			]


	def set_varpar_assignment(self):
		[restaurant.set_varpar_assignment()
			for level in self.tree.itervalues()
				for restaurant in level.itervalues()
				]
				
	def update_varpars(self):
		[vpc.update() for vpc in self.varpar_concents]
		[restaurant.update_varpars()
			for level in self.tree.itervalues()
				for restaurant in level.itervalues()
				]
				
	def get_var_bound(self):
		return (np.sum([restaurant.get_var_bound()
					for level in reversed(self.tree.values())
						for restaurant in level.itervalues()
						])
				+
				np.sum([vpc.get_var_bound() for vpc in self.varpar_concents])
				)


	def iter_bottom_restaurants(self):
		return self.tree[self.n-1].iteritems()

	def get_word_assignment_weights(self, word_ids):
		return self.wrapper.varpar_assignment[word_ids,:]

	def get_word_assignment_weights_missing(self, word_ids):
		return self.wrapper.varpar_assignment_missing[word_ids,:]

class VarParConcent(object):
	def __init__(self, priors):
		self.prior_inv_scale = priors[1]
		self.dps = []
		self.shape = priors[0]
		self.rate = np.random.gamma(1,20)#(self.prior_inv_scale**-1)


	def add_dp(self, dp):
		self.dps.append(dp)
		self.shape+=dp.varpar_stick.shape[0]
		self.mean=self.shape/self.rate
	
	def update(self):
		self.rate = (self.prior_inv_scale
							-
							np.sum(
								[
								dp.E_log_stick[:,1]
								for dp in self.dps
								]
								)
							)
		self.mean=self.shape/self.rate
						
	def get_var_bound(self):
		return -(
				self.shape*np.log(self.rate)
				+
				self.prior_inv_scale*self.mean
				)


class VarParConcent_sublex(object):
	def __init__(self, priors, num_sublex):
		self.prior_rate = priors[1]
		self.dps = []
		self.shape = priors[0]
		self.rate = np.random.gamma(1,20, size=num_sublex)

	def add_dp(self, dp):
		self.dps.append(dp)
		self.shape += dp.varpar_stick.shape[1]
		self.mean = self.shape/self.rate

	def update(self):
		self.rate = (self.prior_rate
							-
							np.sum(
								[
								dp.E_log_stick[...,1]
								for dp in self.dps
								]
								,
								axis=(0, -1)
								)
							)
		self.mean = self.shape/self.rate
						
	def get_var_bound(self):
		return -(
				self.shape*np.sum(np.log(self.rate))
				+
				self.prior_rate*np.sum(self.mean)
				)

class DP_bottom(object):
	def __init__(self, num_clusters, mother, context_length, context, varpar_concent, num_sublex, hdp):
		self.hdp=hdp
		self.mother = mother
		self.context_length=context_length
		self.context = context
		
		self.varpar_concent = varpar_concent

		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_sublex,num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:,:, np.newaxis]

		self.varpar_concent.add_dp(self)

		self.num_clusters = num_clusters
		
		self.data = []
		self.word_ids = []

		self.data_missing = []
		self.word_ids_missing = []

		self.id = self.mother.add_customer(self)
		
		
	def add_customer(self, target_value, word_id, missing=False):
		if missing:
			self.data_missing.append(target_value)
			self.word_ids_missing.append(word_id)
		else:
			self.data.append(target_value)
			self.word_ids.append(word_id)
		
		
	def set_varpar_assignment(self):
		self._update_log_assignment()
		self._update_log_like()


		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(
									self.expected_customer_counts[:,:-1,:] # num_sublex x num_clusters x num_symbols
									,
									axis=-1
									)+1
		self.varpar_stick[...,1] = np.cumsum(
										np.sum(
											self.expected_customer_counts[:,:0:-1,:],
											axis=-1
										)
										,
										axis=1
										)[:,::-1]+self.varpar_concent.mean[:,np.newaxis]
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, :, np.newaxis]
		
	def _update_log_assignment(self):
		appendix = np.zeros((self.E_log_stick.shape[0],1))
		self.log_assignment=(
									np.append(
										self.E_log_stick[...,0],
										appendix
										,
										axis=1
										)[:,:,np.newaxis]
									+np.append(
										appendix,
										np.cumsum(
											self.E_log_stick[...,1]
											,
											axis=1
											)
										,
										axis=1
										)[:,:,np.newaxis]
									+
									self.mother.log_like_top_down[self.id] #  num_sublex x num_clusters x num_symbols
								)

	def _update_log_like(self):
		self.log_like = spm.logsumexp(self.log_assignment, axis=1)


	def update_varpars(self):
		self._update_varpar_stick()
		self._update_log_assignment()
		self._update_log_like()

	def _update_expected_customer_counts(self):
		expected_data_size_per_sublex_and_symbol = np.zeros((num_symbols, self.varpar_stick.shape[0]))
		np.add.at(
				expected_data_size_per_sublex_and_symbol,
				self.data,
				self.hdp.get_word_assignment_weights(self.word_ids) # data_size x num_sublex
				)
		np.add.at(
				expected_data_size_per_sublex_and_symbol,
				self.data_missing,
				self.hdp.get_word_assignment_weights_missing(self.word_ids_missing) # data_size x num_sublex
				)
		self.expected_customer_counts=( #  num_sublex x num_clusters x num_symbols
				expected_data_size_per_sublex_and_symbol.T[:,np.newaxis,:]
				*
				self._get_assign_probs()
				)

	def _get_assign_probs(self):
		return np.exp(
					self.log_assignment #  num_sublex x num_clusters x num_symbols
					-
					spm.logsumexp(self.log_assignment, axis=1)[:,np.newaxis,:]
					)

	def _update_log_like_bottom_up(self):
		self.log_like_bottom_up=self.expected_customer_counts # num_sublex x num_clusters x num_symbols
			
	def get_var_bound(self):
		self._update_expected_customer_counts()
		self._update_log_like_bottom_up()
		return (
						self.get_sum_E_log_p_varpars()
						-self.get_E_log_q_varpar()
						)
		
	def get_sum_E_log_p_varpars(self):
		return np.sum(
						(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,:,1], axis=1) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.




	def get_E_log_q_varpar(self):
		return(
					(
						np.sum(self.E_log_stick*(self.varpar_stick-1))
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							)
						+
						np.sum(sps.gammaln(self.sum_stick))
					) # E[log q(V)]
				)
	
	def set_log_posterior_expectation(self):
		pass
		# self.log_posterior_expectation = spm.logsumexp(np.append(
		# 									np.log(self.varpar_stick[:,0])
		# 									-
		# 									np.log(self.sum_stick)
		# 									,
		# 									0
		# 									)[:,np.newaxis]
		# 								+
		# 								np.append(
		# 									0,
		# 									np.cumsum(
		# 										np.log(self.varpar_stick[:,1])
		# 										-
		# 										np.log(self.sum_stick)
		# 										)
		# 									)[:,np.newaxis]
		# 								+
		# 								self.mother.log_posterior_expectation_top_down[self.id]
		# 								,
		# 								axis=0
		# 								)

	def get_log_posterior_pred(self, value):
		pass
		# return self.log_posterior_expectation[value]






class DP(DP_bottom):
	def __init__(self, num_clusters, mother, context_length, context, varpar_concent, num_sublex, hdp):
		self.mother = mother
		self.hdp=hdp
		self.context_length=context_length
		self.context = context

		self.varpar_concent = varpar_concent
		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_sublex,num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:,:,np.newaxis]

		self.varpar_concent.add_dp(self)

		self.num_clusters = num_clusters
		self.children = []
		self.new_id = 0
		self.id = self.mother.add_customer(self)

	def add_customer(self, child):
		self.children.append(child)
		issued_id = self.new_id
		self.new_id+=1
		return issued_id

	def set_varpar_assignment(self):
		num_children = len(self.children)
		num_tables_child = self.children[0].num_clusters # num_clusters for children.
		self.varpar_assignment = np.random.dirichlet(
											np.ones(self.num_clusters),
											(num_children, self.varpar_stick.shape[0], num_tables_child)
											) # phi in Blei and Jordan.
		assert self.varpar_assignment.size, 'Empty restaurant created.'
		self._update_log_like_top_down()
		
		

	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment[...,:-1], axis=(0, 2))+1
		self.varpar_stick[...,1] = np.cumsum(
										np.sum(
											self.varpar_assignment[...,:0:-1],
											axis=(0, 2)
											)
											,
											axis=-1
											)[...,::-1]+self.varpar_concent.mean[:,np.newaxis]
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:,:,np.newaxis]


	def _update_varpar_assignment(self):
		appendix = np.zeros((self.E_log_stick.shape[0],1))
		self.varpar_assignment = (
									np.append(
										self.E_log_stick[:,:,0],
										appendix
										,
										axis=1
										)[np.newaxis,:,np.newaxis,:]
									+np.append(
										appendix,
										np.cumsum(
											self.E_log_stick[:,:,1]
											,
											axis=1
											)
										,
										axis=1
										)[np.newaxis,:,np.newaxis,:]
									+
									np.sum(
										self.child_log_like_bottom_up[:,:,:,np.newaxis,:] # num_children x num_sublex x data_size x num_symbols
										*
										self.mother.log_like_top_down[self.id,np.newaxis,:,np.newaxis,:,:] #  num_sublex x num_clusters x num_symbols
										,
										axis=-1
										)
								)
		self.varpar_assignment=np.exp(self.varpar_assignment-spm.logsumexp(self.varpar_assignment, axis=-1)[:,:,:,np.newaxis])

	def update_varpars(self):
		self._update_varpar_stick()
		self._update_varpar_assignment()
		self._update_log_like_top_down()

	def _set_child_log_like_bottom_up(self):
		self.child_log_like_bottom_up = np.array(
												[child.log_like_bottom_up 
													for child in self.children
												]
												)
	def get_var_bound(self):
		self._update_log_like_bottom_up()
		return self.get_sum_E_log_p_varpars()-self.get_E_log_q_varpar()



	def _update_log_like_top_down(self):
		self.log_like_top_down = np.sum(
										self.varpar_assignment[:,:,:,:,np.newaxis] 
										*
										self.mother.log_like_top_down[self.id,np.newaxis,:,np.newaxis,:,:] # 1 x num_sublex x 1 x num_clusters x num_symbols
										,
										axis=-2
										)
	
	def _update_log_like_bottom_up(self):
		self._set_child_log_like_bottom_up()
		self.log_like_bottom_up=np.sum(
										self.child_log_like_bottom_up[:,:,:,np.newaxis,:] # num_children x num_sublex x data_size x num_symbols
										*
										self.varpar_assignment[:,:,:,:,np.newaxis]
										,
										axis=(0,2)
										) #  num_sublex x num_clusters x num_symbols

	def get_sum_E_log_p_varpars(self):
		return (
					np.sum(
						(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,:,1], axis=-1) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+
					np.sum(
							self.E_log_stick[:,:,1]*np.cumsum(
														np.sum(
															self.varpar_assignment[...,:0:-1],
															axis=(0,2)
															)
															,
															axis=-1
															)[...,::-1]
							+
							self.E_log_stick[:,:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,2))
							) # E[log p(Z | V)]
					)


	def get_E_log_q_varpar(self):
		return(
					(
						np.sum(self.E_log_stick*(self.varpar_stick-1))
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							)
						+
						np.sum(sps.gammaln(self.sum_stick))
					) # E[log q(V)]
					+
					np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment)) # E[log q(Z)]
					)
	
	def set_log_posterior_expectation(self):
		pass
		# self.log_posterior_expectation = spm.logsumexp(np.append(
		# 									np.log(self.varpar_stick[:,0])
		# 									-
		# 									np.log(self.sum_stick)
		# 									,
		# 									0
		# 									)[:,np.newaxis]
		# 								+
		# 								np.append(
		# 									0,
		# 									np.cumsum(
		# 										np.log(self.varpar_stick[:,1])
		# 										-
		# 										np.log(self.sum_stick)
		# 										)
		# 									)[:,np.newaxis]
		# 								+
		# 								self.mother.log_posterior_expectation_top_down[self.id]
		# 								,
		# 								axis=0
		# 								)
		# self.log_posterior_expectation_top_down = spm.logsumexp(
		# 													np.log(self.varpar_assignment)[:,:,:,np.newaxis] # num_children x data_size x num_clusters
		# 													+
		# 													self.mother.log_posterior_expectation_top_down[self.id,np.newaxis,np.newaxis,:,:] # num_clusters x num_symbols
		# 													,
		# 													axis=-2
		# 												)




	
class DP_top(DP):
	def __init__(self, num_clusters, context_length, context, varpar_concent, atom_base_counts, num_sublex, hdp):
		self.hdp=hdp
		self.context_length=context_length
		self.context = context

		self.varpar_concent = varpar_concent

		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_sublex, num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, :, np.newaxis]

		self.atom_base_counts = atom_base_counts
		self.varpar_atom = np.random.gamma(1, 20, size=(num_sublex, num_clusters, num_symbols))
		self.sum_atom=np.sum(self.varpar_atom, axis=-1)
		self.E_log_atom = (
								sps.digamma(self.varpar_atom)
								-
								sps.digamma(self.sum_atom)[:,:,np.newaxis]
							)
		
		self.varpar_concent.add_dp(self)
		self.num_clusters = num_clusters
		self.new_id=0
		self.children=[]
		
		
	def add_customer(self, child):
		self.children.append(child)
		issued_id = self.new_id
		self.new_id+=1
		return issued_id
		


	def _update_varpar_assignment(self):
		appendix = np.zeros((self.E_log_stick.shape[0],1))
		self.varpar_assignment = (
									np.append(
										self.E_log_stick[:,:,0],
										appendix
										,
										axis=1
										)[np.newaxis,:,np.newaxis,:]
									+np.append(
										appendix,
										np.cumsum(
											self.E_log_stick[:,:,1]
											,
											axis=1
											)
										,
										axis=1
										)[np.newaxis,:,np.newaxis,:]
									+
									np.sum(
										self.child_log_like_bottom_up[:,:,:,np.newaxis,:] # num_children x num_sublex x data_size x num_symbols
										*
										self.E_log_atom[np.newaxis,:,np.newaxis,:,:] #  num_sublex x num_clusters x num_symbols
										,
										axis=-1
										)
								)
		self.varpar_assignment=np.exp(self.varpar_assignment-spm.logsumexp(self.varpar_assignment, axis=-1)[:,:,:,np.newaxis])
		
		


	def _update_varpar_atom(self):
		self.varpar_atom=self.log_like_bottom_up + self.atom_base_counts[np.newaxis,np.newaxis,:]
		self.sum_atom = np.sum(self.varpar_atom, axis=-1)
		self.E_log_atom = (
								sps.digamma(self.varpar_atom)
								-
								sps.digamma(self.sum_atom)[:,:,np.newaxis]
							)
		

	def _update_log_like_top_down(self):
		self.log_like_top_down = np.sum(
										self.varpar_assignment[:,:,:,:,np.newaxis] # num_children x num_sublex x num_child_clusters x num_clusters
										*
										self.E_log_atom[np.newaxis,:,np.newaxis,:,:] # num_sublex x num_clusters x num_symbols
										,
										axis=-2
										)

	

	def update_varpars(self):
		self._update_varpar_stick()
		self._update_varpar_atom()
		self._update_varpar_assignment()
		self._update_log_like_top_down()

	def get_sum_E_log_p_varpars(self):
		return (
					np.sum(
						(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,:,1], axis=-1) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+
					np.sum(
							self.E_log_stick[:,:,1]*np.cumsum(
														np.sum(
															self.varpar_assignment[...,:0:-1],
															axis=(0,2)
															)
															,
															axis=-1
															)[...,::-1]
							+
							self.E_log_stick[:,:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,2))
							) # E[log p(Z | V)]
					+
					np.sum(
						(self.atom_base_counts-1)
						*
						np.sum(self.E_log_atom, axis=(0,1))
						)
					)

	def get_E_log_q_varpar(self):
		return(
					(
						np.sum(self.E_log_stick*(self.varpar_stick-1))
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							)
						+
						np.sum(sps.gammaln(self.sum_stick))
					) # E[log q(V)]
					+
					np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment)) # E[log q(Z)]
					+ # E[log q(U)] below.
					np.sum(
						self.E_log_atom*(self.varpar_atom-1)
					)
					-
					np.sum(
						sps.gammaln(self.varpar_atom)
					)
					+
					np.sum(
						sps.gammaln(self.sum_atom)
					)
					)
	

	def set_log_posterior_expectation(self):
		pass
		# log_expectation_atom = np.log(self.varpar_atom)-np.log(self.sum_atom)[:,np.newaxis]
		# self.log_posterior_expectation = spm.logsumexp(np.append(
		# 													np.log(self.varpar_stick[:,0])
		# 													-
		# 													np.log(self.sum_stick)
		# 													,
		# 													0
		# 													)[:,np.newaxis]
		# 												+
		# 												np.append(
		# 													0,
		# 													np.cumsum(
		# 														np.log(self.varpar_stick[:,1])
		# 														-
		# 														np.log(self.sum_stick)
		# 														)
		# 													)[:,np.newaxis]
		# 												+
		# 												log_expectation_atom
		# 												,
		# 												axis=0
		# 												)
		# self.log_posterior_expectation_top_down = spm.logsumexp(
		# 													np.log(self.varpar_assignment)[:,:,:,np.newaxis] # num_children x data_size x num_clusters
		# 													+
		# 													log_expectation_atom[np.newaxis,np.newaxis,:,:] # num_clusters x num_symbols
		# 													,
		# 													axis=-2
		# 												)

class Beta_Bernoulli(DP_top):
	def __init__(self, priors, data, num_missing_data, num_sublex, wrapper):
		self.priors = np.array(priors) # beta prior

		self.wrapper = wrapper


		self.data = np.zeros((data.size,2))
		self.data[:,0] += (data == 0)
		self.data[:,1] += (data > 0)
		self.data = self.data[:,np.newaxis,:]

		self.varpar_rate = np.random.gamma(
									1, 20,
									size=(num_sublex, 2)
									)
		self.sum_rate = np.sum(self.varpar_rate, axis=-1)
		self.E_log_rate = sps.digamma(self.varpar_rate) - sps.digamma(self.sum_rate)[:,np.newaxis]
		
		self.varpar_categorical = np.random.dirichlet(
										np.ones(2)
										,
										size=num_missing_data
									)

	def _update_varpar_rate(self):
		self.varpar_rate = (
							np.sum(
								self.wrapper.varpar_assignment[:,:,np.newaxis]
								*
								self.data
								,
								axis=0
								)
							+
							np.sum(
								self.wrapper.varpar_assignment_missing[:,:,np.newaxis]
								*
								self.varpar_categorical[:,np.newaxis,:]
								,
								axis=0
							)
							+
							self.priors[np.newaxis,:]
							)
		self.sum_rate = np.sum(self.varpar_rate, axis=-1)
		self.E_log_rate = sps.digamma(self.varpar_rate) - sps.digamma(self.sum_rate)[:,np.newaxis]

	def _update_varpar_categorical(self):
		self.varpar_categorical = np.sum(
									self.wrapper.varpar_assignment_missing[:,:,np.newaxis]
									*
									self.E_log_rate[np.newaxis,:,:]
									,
									axis=1
									)
		self.varpar_categorical = np.exp(self.varpar_categorical - spm.logsumexp(self.varpar_categorical, axis=-1)[:,np.newaxis])

	def update_varpars(self):
		self._update_varpar_rate()
		self._update_varpar_categorical()

	def get_log_like(self):
		return np.sum(
					self.data
					*
					self.E_log_rate[np.newaxis,:,:]
					,
					axis=-1
				)

	def get_log_like_missing(self):
		return np.sum(
				self.varpar_categorical[:,np.newaxis,:]
				*
				self.E_log_rate[np.newaxis,:,:]
				,
				axis=-1
				)

	def get_var_bound(self):
		return self.get_E_log_p() - self.get_E_log_q()

	def get_E_log_p(self):
		return np.sum(
				(self.priors - 1) * np.sum(self.E_log_rate, axis=0)
				)

	def get_E_log_q(self):
		return (
					np.sum(
						self.E_log_rate*(self.varpar_rate-1)
					)
					-
					np.sum(
						sps.gammaln(self.varpar_rate)
					)
					+
					np.sum(
						sps.gammaln(self.sum_rate)
					)
					+
					np.sum(self.varpar_categorical * np.ma.log(self.varpar_categorical))
					)



class Word(object):
	def __init__(self, string, n, word_id, missing=False):
		self.id=word_id
		self.ngrams = [Ngram(window,self)
						for window in zip(*[([num_symbols]*(n-1)+string+[0])[i:] for i in range(n)])
						]
		self.missing = missing
		
		
					
	def get_E_log_likelihoods(self):
		return np.sum(
					[
						ngram.get_E_log_likelihoods() for ngram in self.ngrams
					],
					axis=0
					)
		
	
class Ngram(object):
	def __init__(self, window, word):
		self.word=word
		# self.ids = []
		self.context = window[:-1]
		self.target = window[-1]
		

	def enter_a_restaurant(self, restaurant):
		restaurant.add_customer(self.target, self.word.id, missing=self.word.missing)
		self.restaurant = restaurant
		
	
		
						
	def get_E_log_likelihoods(self):
		return self.restaurant.log_like[...,self.target]

def code_data(training_data,training_data2=None,test_data=None):
	str_training_data=[
						word.split(',')
						for index, word
							in training_data.iteritems()
						]
	if test_data is None:
		str_data=str_training_data[:]
	else:
		str_test_data=[word.split(',') for index, word in test_data.iteritems()]
		str_data=str_training_data+str_test_data
	if not training_data2 is None:
		str_training_data2 = [
								word.split(',')
								for index, word
									in training_data2.iteritems()
								]
		str_data += str_training_data2
	inventory = list(set(itertools.chain.from_iterable(str_data)))


	encoder = {symbol:code for code,symbol in enumerate(inventory, start=1)}
	decoder = {code:symbol for code,symbol in enumerate(inventory, start=1)}

	decoder[0]='END' # Special code reserved for initial symbol.
	decoder[len(decoder)]='START' # Special code reserved for end symbol.

	coded_training_data=[map(lambda s: encoder[s],phrase) for phrase in str_training_data]
	output = (coded_training_data,)
	if not test_data is None:
		coded_test_data=[map(lambda s: encoder[s],phrase) for phrase in str_test_data]
		output += (coded_test_data,)
	if not training_data2 is None:
		coded_training_data2 = [map(lambda s: encoder[s],phrase) for phrase in str_training_data2]
		output += (coded_training_data2,)
	return output + (encoder,decoder)

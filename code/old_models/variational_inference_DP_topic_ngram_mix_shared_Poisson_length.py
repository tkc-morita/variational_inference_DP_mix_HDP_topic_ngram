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
			customers,
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
		logger.info('The base distribution is SHARED %i-gram with DP backoff and Poisson length.' % n)
		logger.info('Results prior to 2017/10/20 have wrong columns on Poisson variational parameters: shape and rate are flipped.')
		
		logger.info('Long vowels and geminates are now (from 03/16/2017) treated as independent segments.')
		logger.info('Script last updated at %s'
							% datetime.datetime.fromtimestamp(
									os.stat(sys.argv[0]).st_mtime
									).strftime('%Y-%m-%d-%H:%M:%S')
							)
		
		global num_symbols
		if inventory_size is None:
			num_symbols = len(set(reduce(lambda x,y: x+y, customers)))
		else:
			num_symbols = inventory_size
		logger.info('# of symbols: %i' % num_symbols)

		dirichlet_base_counts = dirichlet_concentration * np.ones(num_symbols)
		
		
		self.customers = [Word(word, n, id) for id,word in enumerate(customers)]
		num_customers = len(customers)
		logger.info('# of words: %i' % num_customers)

		word_lengths = np.array([len(word) for word in customers])
		mean_length = np.mean(word_lengths)

		
		self.varpar_assignment = np.random.dirichlet(np.ones(num_sublex), num_customers) # phi in Blei and Jordan.
		
		
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
								for word in self.customers
								for ngram in word.ngrams
								])

		self.hdp_ngram = HDPNgram(T_base, n, concent_priors, dirichlet_base_counts, full_contexts)
		
		self.sublex_ngrams=SublexNgram(num_sublex, T_base, n, concent_priors, self.customers, full_contexts, self)

		self.poisson_length = Poisson_length((mean_length,1), word_lengths, num_sublex, self)
		
		self.hdp_ngram.set_varpar_assignment()
		self.sublex_ngrams.set_varpar_assignment()
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
		# log_pred_prob_list=[]
		# log_stick_weights=(
		# 						np.append(
		# 							np.log(self.varpar_stick[:,0])
		# 							-
		# 							np.log(self.sum_stick)
		# 							,
		# 							0
		# 							)
		# 						+
		# 						np.append(
		# 							0,
		# 							np.cumsum(
		# 								np.log(self.varpar_stick[:,1])
		# 								-
		# 								np.log(self.sum_stick)
		# 								)
		# 							)
		# 					)
		# [sublex.set_log_posterior_expectation() for seblex in self.sublex_ngrams]
		# for string in test_data:
		# 	test_word = Word(string, self.n, -1)
		# 	log_pred_prob=spm.logsumexp(
		# 					log_stick_weights
		# 					+
		# 					np.array(
		# 						[sublex.get_log_posterior_pred_per_word(test_word)
		# 						for sublex in self.sublex_ngrams
		# 						]
		# 						)
		# 					)
		# 	log_pred_prob_list.append(log_pred_prob)
		# return log_pred_prob_list
		
	def save_results(self, decoder):

		inventory=[('symbol_code_%i' % code) for code in decoder.keys()]#[symbol.encode('utf-8') for symbol in decoder.values()]
		# Shared HDP ngram
		with pd.HDFStore(
				os.path.join(self.result_path,'variational_parameters.h5')
# 				,
# 				encoding='utf-8'
				) as hdf5_store:
			for context_length,level in self.hdp_ngram.tree.iteritems():
				for context,rst in level.iteritems():
					coded_context = '_'.join(map(str,context))#.encode('utf-8')

					if context_length == self.n-1:
						df_assignment = pd.DataFrame(
										rst.varpar_assignment.flatten()[:,np.newaxis]
										,
										columns=["p"]
										)
						num_sublex = rst.varpar_assignment.shape[0]
						num_child_clusters = rst.varpar_assignment.shape[1]
						num_clusters = rst.varpar_assignment.shape[2]
						df_assignment['sublex_id'] = np.repeat(
																np.arange(num_sublex)
																,
																num_child_clusters*num_clusters
															)
						df_assignment['children_cluster_id']=np.tile(
																np.repeat(
																	np.arange(num_child_clusters)
																	,
																	num_clusters
																	)
																,
																num_sublex
																)
						df_assignment['cluster_id']=np.tile(
														np.arange(num_clusters)
														,
														num_sublex
														*
														num_child_clusters
													)
					else:
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
																	rst.varpar_assignment.shape[1]
																	*
																	rst.varpar_assignment.shape[2]
																	)
																].reset_index(drop=True)
						df_assignment['children_cluster_id']=np.tile(
																np.repeat(
																	np.arange(rst.varpar_assignment.shape[1])
																	,
																	rst.varpar_assignment.shape[2]
																	)
																,
																rst.varpar_assignment.shape[0]
																)
						df_assignment['cluster_id']=np.tile(
														np.arange(rst.varpar_assignment.shape[2])
														,
														rst.varpar_assignment.shape[0]
														*
														rst.varpar_assignment.shape[1]
													)
					hdf5_store.put(
						("shared/_%igram/context_%s/assignment"
						% (context_length+1,coded_context))
						,
						df_assignment
						,
# 						encoding="utf-8"
						)
			
					df_stick=pd.DataFrame(rst.varpar_stick, columns=('beta_par1','beta_par2'))
					df_stick['cluster_id']=df_stick.index
					hdf5_store.put(
						("shared/_%igram/context_%s/stick"
						% (context_length+1,coded_context))
						,
						df_stick
						,
# 						encoding="utf-8"
						)
			
					
				
			df_concent = pd.DataFrame(
								[(vpc.shape, vpc.rate)
								for vpc
								in self.hdp_ngram.varpar_concents]
								,
								columns=['shape', 'rate']
							)
			df_concent['context_length']=self.hdp_ngram.tree.keys()
			hdf5_store.put(
							"shared/concentration",
							df_concent,
# 							encoding='utf-8'
							)

			df_atom = pd.DataFrame(
						self.hdp_ngram.tree[0][()].varpar_atom.flatten()[:,np.newaxis],
						columns=['dirichlet_par']
						)
			atom_shape = self.hdp_ngram.tree[0][()].varpar_atom.shape
			df_atom['cluster_id']=np.repeat(
										np.arange(atom_shape[0])
										,
										atom_shape[1]
									)
			df_atom['value']=pd.Series(
								np.tile(
									np.arange(atom_shape[1])
									,
									atom_shape[0]
								)
								)
			
			hdf5_store.put(
							'shared/_1gram/context_/atom'
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
			df_assignment_sl.to_csv(os.path.join(self.result_path, "SubLexica_assignment.csv"), index=False, encoding='utf-8')
			hdf5_store.put(
							"sublex/assignment",
							df_assignment_sl,
# 							encoding='utf-8'
							)
		
			df_stick_sl = pd.DataFrame(self.varpar_stick, columns=('beta_par1','beta_par2'))
			df_stick_sl['cluster_id']=df_stick_sl.index
			hdf5_store.put(
							"sublex/stick",
							df_stick_sl,
							)
		
			df_concent=pd.DataFrame(
						self.sublex_ngrams.varpar_concent_bottom.rate[:,np.newaxis]
						,
						columns=['rate']
						)
			df_concent['shape'] = self.sublex_ngrams.varpar_concent_bottom.shape
			df_concent['DP_name'] = [('%igram_%i' % (self.n,sublex_id))
										for sublex_id
										in xrange(self.num_clusters)
										]
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
			
			df_length = pd.DataFrame(self.poisson_length.varpar_length, columns=['shape','rate'])
			df_length['sublex_id'] = df_length.index
			hdf5_store.put(
							"sublex/length",
							df_length,
							)

			
		

			for context,bottom_rst in self.sublex_ngrams.restaurants.iteritems():
				coded_context = '_'.join(map(str,context))
				num_sublex = bottom_rst.varpar_stick.shape[0]
				num_clusters = bottom_rst.varpar_stick.shape[1]
				df_stick=pd.DataFrame(bottom_rst.varpar_stick.reshape(
											num_sublex*num_clusters
											,
											2
										),
										columns=('beta_par1','beta_par2')
										)
				df_stick['sublex_id'] = np.repeat(np.arange(num_sublex), num_clusters)
				df_stick['cluster_id'] = np.tile(np.arange(num_clusters), num_sublex)

				hdf5_store.put(
					("sublex/context_%s/stick"
					% coded_context)
					,
					df_stick
					,
# 						encoding='utf-8'
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
		log_varpar_assignment = (
									np.append(
										self.E_log_stick[:,0],
										0
										)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.E_log_stick[:,1]
											)
										)[np.newaxis,:]
									+
									self.word_likelihood
									)
		self.varpar_assignment=np.exp(log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,np.newaxis])

	def _update_word_likelihood(self):
		self.word_likelihood = (
									np.array(
										[
											word.get_E_log_likelihoods() # Output an array of length num_sublex
												for word in self.customers
										]
										)
									+
									self.poisson_length.get_log_like()
									)

	def update_varpars(self):
		self.varpar_concent.update()
		self._update_varpar_stick()
		self.hdp_ngram.update_varpars()
		self.sublex_ngrams.update_varpars()
		self.poisson_length.update_varpars()
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
				self.sublex_ngrams.get_var_bound()
				+
				self.hdp_ngram.get_var_bound()
				+
				self.poisson_length.get_var_bound()
				+
				self.get_E_log_p()
				-
				self.get_E_log_q()
				)

	def get_E_log_p(self):
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
													)[::-1]
					+
					self.E_log_stick[:,0]*np.sum(self.varpar_assignment[:,:-1], axis=0)
					) # E[log p(Z | V)]
				+
				np.sum(
					self.word_likelihood # num_words x num_sublex
					*
					self.varpar_assignment
					) # E[log p(X | Z,eta)]
				) # E[log p(V, alpha)]


	def get_E_log_q(self):
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
				)
		

class SublexNgram(object):
	def __init__(self, num_sublex, num_clusters, n, concent_priors, customers, full_contexts, wrapper):
		self.wrapper=wrapper
		self.n = n


		self.varpar_concent_bottom = VarParConcent_bottom(concent_priors, num_sublex)


		self.restaurants={context:
							DP_bottom(
									num_clusters,
									wrapper.hdp_ngram.tree[n-1][context],
									n-1,
									context,
									self.varpar_concent_bottom,
									num_sublex,
									self
									)
							for context in full_contexts
							}

		[ngram.enter_a_restaurant(self.restaurants[ngram.context])
			for word in customers
			for ngram in word.ngrams
			]
		
	def get_word_assignment_weights(self, word_ids):
		return self.wrapper.varpar_assignment[word_ids,:]
	
	def set_log_posterior_expectation(self):
		[restaurant.set_log_posterior_expectation()
					for level in self.tree.itervalues()
						for restaurant in level.itervalues()
						]

	def get_log_posterior_pred_per_word(self, test_word):
		log_pred_prob=np.float64(0)
		for ngram in test_word.ngrams:
			for minus_conlen,context_length in enumerate(reversed(xrange(self.n))):
				sub_context=ngram.context[minus_conlen:]
				if sub_context in self.tree[context_length].iterkeys():
					restaurant=self.tree[context_length][sub_context]
					log_pred_prob+=restaurant.get_log_posterior_pred(ngram.target)
					break
		return log_pred_prob


	
	
	def set_varpar_assignment(self):
		[restaurant.set_varpar_assignment()
			for restaurant in self.restaurants.itervalues()
				]
				
	def update_varpars(self):
		self.varpar_concent_bottom.update()
		[restaurant.update_varpars()
			for restaurant in self.restaurants.itervalues()
				]
				
	def get_var_bound(self):
		return (np.sum([restaurant.get_var_bound()
					for restaurant in self.restaurants.itervalues()
					])
				+
				self.varpar_concent_bottom.get_var_bound()
				)

		
class HDPNgram(object):
	def __init__(self, num_clusters, n, concent_priors, dirichlet_base_counts, full_contexts):
# 		self.wrapper=wrapper
		self.n = n
		self.tree = {k:{} for k in xrange(n)}
		self.varpar_concents = [VarParConcent(concent_priors) for context_length in xrange(n)]
		
		self.tree[0][()]=DP_top(
							num_clusters,
							0,
							(),
							self.varpar_concents[0],
							dirichlet_base_counts,
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
									self
									)

		for context in full_contexts:
			self.tree[self.n-1][context]\
				= DP_shared_bottom(
					num_clusters,
					self.tree[n-2][context[1:]],
					n-1,
					context,
					self.varpar_concents[-1],
					self
					)


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

class VarParConcent(object):
	def __init__(self, priors):
		self.prior_rate = priors[1]
		self.dps = []
		self.shape = priors[0]
		self.rate = np.random.gamma(1,20)#(self.prior_rate**-1)


	def add_dp(self, dp):
		self.dps.append(dp)
		self.shape+=dp.varpar_stick.shape[0]
		self.mean=self.shape/self.rate
	
	def update(self):
		self.rate = (self.prior_rate
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
				self.prior_rate*self.mean
				)

class VarParConcent_bottom(VarParConcent):
	def __init__(self, priors, num_sublex):
		self.prior_rate = priors[1]
		self.dps = []
		self.shape = priors[0]
		self.rate = np.random.gamma(1,20, size=num_sublex)

	def add_dp(self, dp):
		self.dps.append(dp)
		self.shape+=dp.varpar_stick.shape[1]
		self.mean=self.shape/self.rate

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
		self.mean=self.shape/self.rate
						
	def get_var_bound(self):
		return -(
				self.shape*np.sum(np.log(self.rate))
				+
				self.prior_rate*np.sum(self.mean)
				)

class DP_bottom(object):
	def __init__(self, num_clusters, mother, context_length, context, varpar_concent, num_sublex, sublex_wrappter):
		self.sublex_wrappter=sublex_wrappter
		self.mother = mother
		self.context_length=context_length
		self.context = context
		
		self.varpar_concent = varpar_concent

		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_sublex, num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:,:, np.newaxis]

		self.varpar_concent.add_dp(self)

		self.num_clusters = num_clusters
		self.new_id = 0
		self.customers=[]
		self.word_ids=[]
		self.id = self.mother.add_customer(self)
		
		
	def add_customer(self, target_value, word_id):
		issued_id = self.new_id
		self.new_id+=1
		self.customers.append(target_value)
		self.word_ids.append(word_id)
		return issued_id
		
		
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
									self.mother.log_like_top_down # num_sublex x num_clusters x num_symbols
								)

	def _update_log_like(self):
		self.log_like = spm.logsumexp(self.log_assignment, axis=1)


	def update_varpars(self):
		self._update_varpar_stick()
		self._update_log_assignment()
		self._update_log_like()

	def _update_expected_customer_counts(self):
		expected_customers_per_sublex_and_symbol = np.zeros((num_symbols, self.varpar_stick.shape[0]))
		np.add.at(
				expected_customers_per_sublex_and_symbol,
				self.customers,
				self.sublex_wrappter.get_word_assignment_weights(self.word_ids) # data_size x num_sublex
				)
		self.expected_customer_counts=( # num_sublex x num_clusters x num_symbols
				expected_customers_per_sublex_and_symbol.T[:,np.newaxis,:]
				*
				self._get_assign_probs()
				)

	def _get_assign_probs(self):
		return np.exp(
					self.log_assignment # num_sublex x num_clusters x num_symbols
					-
					spm.logsumexp(self.log_assignment, axis=1)[:,np.newaxis,:]
					)

	def _update_log_like_bottom_up(self):
		self.log_like_bottom_up=self.expected_customer_counts # num_sublex x num_clusters x num_symbols
			
	def get_var_bound(self):
		self._update_expected_customer_counts()
		self._update_log_like_bottom_up()
		return (
						self.get_E_log_p()
						-self.get_E_log_q()
						)
		
	def get_E_log_p(self):
		return np.sum(
						(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,:,1], axis=1) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.




	def get_E_log_q(self):
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

class DP_shared_bottom(DP_bottom):
	def __init__(self, num_clusters, mother, context_length, context, varpar_concent, hdp):
		self.mother = mother
		self.hdp=hdp
		self.context_length=context_length
		self.context = context

		self.varpar_concent = varpar_concent
		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, np.newaxis]

		self.varpar_concent.add_dp(self)

		self.num_clusters = num_clusters
		self.id = self.mother.add_customer(self)

	def add_customer(self, child):
		self.child = child
		return 0

	def set_varpar_assignment(self):
		num_sublex = self.child.varpar_stick.shape[0]
		num_tables_child = self.child.num_clusters
		self.varpar_assignment = np.random.dirichlet(
											np.ones(self.num_clusters),
											(
												num_sublex,
												num_tables_child,
												)
											) # phi in Blei and Jordan.
		assert self.varpar_assignment.size, 'Empty restaurant created.'
		self._update_log_like_top_down()

	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment[...,:-1], axis=(0, 1))+1
		self.varpar_stick[...,1] = np.cumsum(
										np.sum(
											self.varpar_assignment[...,:0:-1],
											axis=(0, 1)
											)
											)[::-1]+self.varpar_concent.mean
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, np.newaxis]

	def _update_varpar_assignment(self):
		log_varpar_assignment = (
									np.append(
										self.E_log_stick[:,0],
										0
										)[np.newaxis,np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.E_log_stick[:,1]
											)
										)[np.newaxis,np.newaxis,:]
									+
									np.sum(
										self.child_log_like_bottom_up[:,:,np.newaxis,:] # num_children x num_customers x num_symbols
										*
										self.mother.log_like_top_down[np.newaxis,np.newaxis,self.id,:,:] #  num_clusters x num_symbols
										,
										axis=-1
										)
								)
		self.varpar_assignment=np.exp(log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,:,np.newaxis])

	def update_varpars(self):
		self._update_varpar_stick()
		self._update_varpar_assignment()
		self._update_log_like_top_down()

	def _set_child_log_like_bottom_up(self):
		self.child_log_like_bottom_up = self.child.log_like_bottom_up # num_customers x num_symbols


	def get_var_bound(self):
		self._update_log_like_bottom_up()
		return self.get_E_log_p()-self.get_E_log_q()



	def _update_log_like_top_down(self):
		self.log_like_top_down = np.sum(
										self.varpar_assignment[:,:,:,np.newaxis] 
										*
										self.mother.log_like_top_down[np.newaxis,np.newaxis,self.id,:,:] # num_clusters x num_symbols
										,
										axis=-2
										)

	def _update_log_like_bottom_up(self):
		self._set_child_log_like_bottom_up()
		self.log_like_bottom_up=np.sum(
										self.child_log_like_bottom_up[:,:,np.newaxis,:] # num_sublex x num_customers x num_symbols
										*
										self.varpar_assignment[:,:,:,np.newaxis]
										,
										axis=(0,1)
										) #  num_clusters x num_symbols



	def get_E_log_p(self):
		return (
					(
						(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,1]) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+
					np.sum(
							self.E_log_stick[:,1]*np.cumsum(
														np.sum(
															self.varpar_assignment[...,:0:-1],
															axis=(0,1)
															)
															)[::-1]
							+
							self.E_log_stick[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
							) # E[log p(Z | V)]
					)


	def get_E_log_q(self):
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


class DP(DP_shared_bottom):
	def __init__(self, num_clusters, mother, context_length, context, varpar_concent, hdp):
		self.mother = mother
		self.hdp=hdp
		self.context_length=context_length
		self.context = context

		self.varpar_concent = varpar_concent
		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, np.newaxis]

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
											(num_children, num_tables_child)
											) # phi in Blei and Jordan.
		assert self.varpar_assignment.size, 'Empty restaurant created.'
		self._update_log_like_top_down()
		
		
	def _set_child_log_like_bottom_up(self):
		self.child_log_like_bottom_up = np.array(
												[child.log_like_bottom_up 
													for child in self.children
												]
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
		# 													np.log(self.varpar_assignment)[:,:,:,np.newaxis] # num_children x num_customers x num_clusters
		# 													+
		# 													self.mother.log_posterior_expectation_top_down[self.id,np.newaxis,np.newaxis,:,:] # num_clusters x num_symbols
		# 													,
		# 													axis=-2
		# 												)




	
class DP_top(DP):
	def __init__(self, num_clusters, context_length, context, varpar_concent, atom_base_counts, hdp):
		self.hdp=hdp
		self.context_length=context_length
		self.context = context

		self.varpar_concent = varpar_concent

		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(num_clusters-1,2)
												) # gamma in Blei and Jordan.
		self.sum_stick = np.sum(self.varpar_stick, axis=-1)
		self.E_log_stick = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_stick)[:, np.newaxis]

		self.atom_base_counts = atom_base_counts

		self.varpar_atom = np.random.gamma(1, 20, size=(num_clusters, num_symbols))
		self.sum_atom=np.sum(self.varpar_atom, axis=1)
		self.E_log_atom = (
								sps.digamma(self.varpar_atom)
								-
								sps.digamma(self.sum_atom)[:,np.newaxis]
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
		log_varpar_assignment = (
									np.append(
										self.E_log_stick[:,0],
										0
										)[np.newaxis,np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.E_log_stick[:,1]
											)
										)[np.newaxis,np.newaxis,:]
									+
									np.sum(
										self.child_log_like_bottom_up[:,:,np.newaxis,:] # num_children x num_customers x num_symbols
										*
										self.E_log_atom[np.newaxis,np.newaxis,:,:] #  num_clusters x num_symbols
										,
										axis=-1
										)
								)
		self.varpar_assignment=np.exp(log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,:,np.newaxis])
		
		


	def _update_varpar_atom(self):
		self.varpar_atom=self.log_like_bottom_up + self.atom_base_counts[np.newaxis,:]
		self.sum_atom = np.sum(self.varpar_atom, axis=-1)
		self.E_log_atom = (
								sps.digamma(self.varpar_atom)
								-
								sps.digamma(self.sum_atom)[:,np.newaxis]
							)
		

	def _update_log_like_top_down(self):
		self.log_like_top_down = np.sum(
										self.varpar_assignment[:,:,:,np.newaxis]
										*
										self.E_log_atom[np.newaxis,np.newaxis,:,:]
										,
										axis=-2
										)

	

	def update_varpars(self):
		self._update_varpar_stick()
		self._update_varpar_atom()
		self._update_varpar_assignment()
		self._update_log_like_top_down()

	def get_E_log_p(self):
		return (
					(
						(self.varpar_concent.mean-1)*np.sum(self.E_log_stick[:,1]) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+
					np.sum(
							self.E_log_stick[:,1]*np.cumsum(
														np.sum(
															self.varpar_assignment[...,:0:-1],
															axis=(0,1)
															)
															)[::-1]
							+
							self.E_log_stick[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
							) # E[log p(Z | V)]
					+
					np.sum(
						(self.atom_base_counts-1)
						*
						np.sum(self.E_log_atom, axis=0)
					)
					)

	def get_E_log_q(self):
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
		# 													np.log(self.varpar_assignment)[:,:,:,np.newaxis] # num_children x num_customers x num_clusters
		# 													+
		# 													log_expectation_atom[np.newaxis,np.newaxis,:,:] # num_clusters x num_symbols
		# 													,
		# 													axis=-2
		# 												)

class Poisson_length(DP_top):
	def __init__(self, priors, data, num_sublex, wrapper):
		self.prior_shape = priors[0]
		self.prior_rate = priors[1]
		
		self.wrapper = wrapper

		self.varpar_length = np.random.gamma(
									1, 20,
									size=(num_sublex, 2)
									)
		self.E_length = self.varpar_length[:,0] / self.varpar_length[:,1]
		self.E_log_length = sps.digamma(self.varpar_length[:,0]) - np.log(self.varpar_length[:,1])

		self.data = data



	def _update_varpar_length(self):
		self.varpar_length[:,0] = self.sum_expected_length + self.prior_shape
		self.varpar_length[:,1] = self.expected_num_obs + self.prior_rate
		self.E_length = self.varpar_length[:,0] / self.varpar_length[:,1]
		self.E_log_length = sps.digamma(self.varpar_length[:,0]) - np.log(self.varpar_length[:,1])

	def get_log_like(self):
		return (
				self.data[:,np.newaxis]
				*
				self.E_log_length[np.newaxis,:]
				-
				self.E_length[np.newaxis,:]
				)

	def update_varpars(self):
		self._update_expected_counts()
		self._update_varpar_length()

	def _update_expected_counts(self):
		self.sum_expected_length = np.sum(
										self.wrapper.varpar_assignment
										*
										self.data[:,np.newaxis]
										,
										axis=0
									)
		self.expected_num_obs = np.sum(self.wrapper.varpar_assignment, axis=0)

	def get_var_bound(self):
		return self.get_E_log_p() - self.get_E_log_q()

	def get_E_log_p(self):
		return np.sum(
				(self.prior_shape - 1) * self.E_log_length
				-
				self.prior_rate * self.E_length
				)

	def get_E_log_q(self):
		return np.sum(
					self.varpar_length[:,0] * np.log(self.varpar_length[:,1])
					-
					sps.gammaln(self.varpar_length[:,0])
					+
					(self.varpar_length[:,0] - 1) * self.E_log_length
					-
					self.varpar_length[:,0]
					)


class Word(object):
	def __init__(self, string, n, id):
		self.id=id
		self.ngrams = [Ngram(window,self)
						for window in zip(*[([num_symbols]*(n-1)+string)[i:] for i in range(n)])
						]
		
		
					
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
		restaurant.add_customer(self.target, self.word.id)
		self.restaurant = restaurant
		
	
		
						
	def get_E_log_likelihoods(self):
		return self.restaurant.log_like[...,self.target]




def code_data(training_data,test_data=None):
	str_training_data=[word.split(',') for index, word in training_data.iteritems()]
	if test_data is None:
		str_data=str_training_data
	else:
		str_test_data=[word.split(',') for index, word in test_data.iteritems()]
		str_data=str_training_data+str_test_data
	inventory = list(set(itertools.chain.from_iterable(str_data)))


	encoder = {symbol:code for code,symbol in enumerate(inventory)}
	decoder = {code:symbol for code,symbol in enumerate(inventory)}

	decoder[len(decoder)]='START' # Special code reserved for end symbol.

	if test_data is None:
		coded_data = [map(lambda s: encoder[s],phrase) for phrase in str_data]
		return (coded_data,encoder,decoder)
	else:
		coded_training_data=[map(lambda s: encoder[s],phrase) for phrase in str_training_data]
		coded_test_data=[map(lambda s: encoder[s],phrase) for phrase in str_test_data]
		return (coded_training_data,coded_test_data,encoder,decoder)


if __name__=='__main__':
# 	warnings.simplefilter('error', UserWarning)
	datapath = sys.argv[1]
	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')
# 	str_data = list(df.IPA_)
# 	with open(datapath,'r') as f: # Read the phrase file.
# 		str_data = [phrase.replace('\r','\n').strip('\n').split('\t')[0]
# 								for phrase in f.readlines()
# 								]
	customers,encoder,decoder = code_data(df.IPA_csv)
	now = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
	result_path = os.path.join(
					# '/om/user/tmorita/results_shared_base/',
					'results',
					os.path.splitext(datapath.split('/')[-1])[0],
					now
					)
	# result_path = ('./results/'+os.path.splitext(datapath.split('/')[-1])[0]+'_'+now+'/')
	os.makedirs(result_path)
	num_sublex = int(sys.argv[2])
	n = int(sys.argv[3])
	T_base = len(decoder)*2 # Number of symbols x 2
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
	dirichlet_concentration = np.float64(1)
	max_iters = int(sys.argv[4])
	min_increase = np.float64(sys.argv[5])
	start = datetime.datetime.now()
	vi = VariationalInference(
			num_sublex,
			customers,
# 			sl_concent,
			n,
			T_base,
			concent_priors,
			dirichlet_concentration,
			result_path
			)
	vi.train(max_iters, min_increase)
	vi.save_results(decoder)
	print 'Time spent',str(datetime.datetime.now()-start)
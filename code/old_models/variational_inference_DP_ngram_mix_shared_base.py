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
			T,
			customers,
# 			concent_priors,
			n,
			T_base,
			concent_priors,
# 			noise,
			result_path,
			entire_inventory_size=None
			):
# 		self.setup_logger(result_path)
		update_log_handler(result_path)
		
		logger.info('DP mixture of words.')
		self.n=n
		logger.info('The base distribution is SHARED %i-gram with DP backoff.' % n)
		
		logger.info('Long vowels and geminates are now (from 03/16/2017) treated as independent segments.')
		logger.info('Script last updated at %s'
							% datetime.datetime.fromtimestamp(
									os.stat(sys.argv[0]).st_mtime
									).strftime('%Y-%m-%d-%H:%M:%S')
							)
		
		global num_symbols
		if entire_inventory_size is None:
			num_symbols = len(set(reduce(lambda x,y: x+y, customers)))+2
		else:
			num_symbols = entire_inventory_size
		logger.info('# of symbols: %i' % num_symbols)
		
		global log_error_mat
		noise = np.finfo(np.float64).epsneg # Smallest representable positive number s.t. 1-noise!=1
		noise_per_symbol = noise/(num_symbols-1)
		assert noise_per_symbol, 'noise/(num_symbols-1) is to small to represent.'
		log_error_mat = np.full((num_symbols,)*2, noise_per_symbol)
		np.fill_diagonal(
						log_error_mat,
						1-noise
						)
		log_error_mat = np.log(log_error_mat)
		logger.info('Noise of transmission from tables to customers: %f' % noise)
# 		print np.exp(log_error_mat).sum(axis=1)
		
		self.customers = [Word(word, n, id) for id,word in enumerate(customers)]
		num_customers = len(customers)
		logger.info('# of words: %i' % num_customers)
		
		self.varpar_assignment = np.random.dirichlet(np.ones(T), num_customers) # phi in Blei and Jordan.
		
		
		self.concent_priors = concent_priors
		
		self.varpar_concent=np.zeros(2)
		self.varpar_concent[0] = self.concent_priors[0]+T-1 # The shape of a gamma distr.
		self.varpar_concent[1] = np.random.gamma(
												1,
												#self.concent_priors[1]**-1
												20
												) # The INVERSE scale of a gamma distr.
		self.concent_mean=self.varpar_concent[0]/self.varpar_concent[1]
		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		self.varpar_stick = np.random.gamma(1,#self.concent_priors[0],
												20,#self.concent_priors[1],
												size=(T-1,2)
												) # gamma in Blei and Jordan.
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]
		self.T = T
		logger.info('(max) # of tables for sublexicalization: %i' % T)
		
		full_contexts = set([ngram.context
								for word in self.customers
								for ngram in word.ngrams
								])

		self.hdp_ngram = HDPNgram(T_base, n, concent_priors, 0, full_contexts)
		
		self.sublex_ngrams=[SublexNgram(T_base, n, concent_priors, id, self.customers, full_contexts, self)
								for id in xrange(self.T)
								]
		
		self.hdp_ngram.set_varpar_assignment()
		[sublex.set_varpar_assignment() for sublex in self.sublex_ngrams]
		
		logger.info('# of tables: %i' % T_base)
		logger.info('Gamma priors on concent_priorsration: (%f,%f)'
							% (concent_priors[0],concent_priors[1]**-1)
							)

		self.result_path = result_path 
		logger.info('Initialization complete.')


		
	
		
	
	def train(self, max_iters, min_increase):
		logger.info("Main loop started.")
		logger.info("Max iteration is %i." % max_iters)
		logger.info("Will be terminated if variational bound is only improved by <=%f." % min_increase)
		converged=False
		iter_id=0
# 		self.sum_E_log_p_eta_lambda=-np.inf
# 		self.E_log_varpar=-np.inf
# 		self.E_log_likelihoods=-np.inf
		self.current_var_bound = self.get_var_bound()
# 		self.intermediate_var_bound=self.current_var_bound
		while iter_id<max_iters:
			iter_id+=1
			self.update_varpars()
			logger.info("Variational parameters updated (Iteration ID: %i)." % iter_id)
			new_var_bound = self.get_var_bound()
			improvement = new_var_bound-self.current_var_bound
			logger.info("Current var_bound is %0.12f (%+0.12f)." % (new_var_bound,improvement))
# 			print improvement
			if np.isnan(new_var_bound):
# 				self.save_results(decoder,inventory)
				raise Exception("nan detected.")
			if improvement<0:
				logger.error("variational bound decreased. Something wrong.")
# 				self.save_results(decoder,inventory)
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
		log_pred_prob_list=[]
		log_stick_weights=(
								np.append(
									np.log(self.varpar_stick[:,0])
									-
									np.log(self.sum_gamma)
									,
									0
									)
								+
								np.append(
									0,
									np.cumsum(
										np.log(self.varpar_stick[:,1])
										-
										np.log(self.sum_gamma)
										)
									)
							)
		for string in test_data:
			test_word = Word(string, self.n, -1)
			log_pred_prob=spm.logsumexp(
							log_stick_weights
							+
							np.array(
								[sublex.get_log_posterior_pred_per_word(test_word)
								for sublex in self.sublex_ngrams
								]
								)
							)
			log_pred_prob_list.append(log_pred_prob)
		return log_pred_prob_list
		
	def save_results(self, decoder):
		
		inventory=[('symbol_code_%i' % code) for code in decoder.keys()]#[symbol.encode('utf-8') for symbol in decoder.values()]
# 		print inventory
		# Shared HDP ngram
		with pd.HDFStore(
				os.path.join(self.result_path,'variational_parameters.h5')
# 				,
# 				encoding='utf-8'
				) as hdf5_store:
			for context_length,level in self.hdp_ngram.tree.iteritems():
				for context,rst in level.iteritems():
					# coded_context = '_'.join([decoder[code] for code in context])#.encode('utf-8')
# 					children_contexts = ['_'.join([decoder[code] for code in child_dp.context])
# 											for child_dp in rst.children]
					coded_context = '_'.join(map(str,context))#.encode('utf-8')
					children_contexts = pd.Series(['_'.join([str(code) for code in child_dp.context])
											for child_dp in rst.children])

					df_assignment = pd.DataFrame(
									rst.varpar_assignment.flatten()[:,np.newaxis]
									,
									columns=["p"]
									)
# 					df_assignment = pd.DataFrame(rst.varpar_assignment.reshape(
# 																		rst.num_children*rst.num_tables_child,
# 																		rst.T
# 																		)
# 												,
# 												columns=[
# 													('cluster_id_%i' % table_id)
# 													for table_id in range(rst.T)
# 													]
# 												)
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
					# df_assignment = pd.DataFrame(rst.varpar_assignment.reshape(
					# 													rst.num_children*rst.num_tables_child,
					# 													rst.T
					# 													)
					# 							,
					# 							columns=[("cluster_id_%i" % table_id)
					# 										for table_id in range(rst.T)]
					# 							)
					# df_assignment['children_DP_context']=np.repeat(children_contexts, rst.num_tables_child)
					# df_assignment['children_cluster_id']=np.tile(np.arange(rst.num_tables_child), rst.num_children)
# 					df_assignment.to_csv(self.result_path
# 											+("shared-%igram_context-%s_assignment.csv" % (context_length+1,coded_context)),
# 											index=False,
# 											encoding='utf-8')
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
# 					df_stick.to_csv(self.result_path
# 									+("shared-%igram_context-%s_stick.csv"
# 										% (context_length+1,coded_context)),
# 									index=False,
# 									encoding='utf-8')
					hdf5_store.put(
						("shared/_%igram/context_%s/stick"
						% (context_length+1,coded_context))
						,
						df_stick
						,
# 						encoding="utf-8"
						)
			
					df_label=pd.DataFrame(
								rst.varpar_label.flatten()[:,np.newaxis],
								columns=["p"]
								)
					df_label['cluster_id']=np.repeat(
												np.arange(rst.varpar_label.shape[0])
												,
												rst.varpar_label.shape[1]
											)
					df_label['value']=pd.Series(
										np.tile(
											np.arange(rst.varpar_label.shape[1])
											,
											rst.varpar_label.shape[0]
										)
										)
					# df_label=pd.DataFrame(rst.varpar_label, columns=inventory)
					# df_label['cluster_id']=df_label.index
# 					df_label.to_csv(self.result_path
# 										+("shared-%igram_context-%s_label.csv"
# 											% (context_length+1,coded_context)),
# 											index=False,
# 											encoding='utf-8')
					hdf5_store.put(
						("shared/_%igram/context_%s/label"
						% (context_length+1,coded_context))
						,
						df_label
						,
# 						encoding="utf-8"
						)
				
			df_concent = pd.DataFrame(self.hdp_ngram.varpar_concent_shape[:,np.newaxis],
										columns=['shape']
										)
			df_concent['scale']=self.hdp_ngram.varpar_concent_inv_scale**-1
			df_concent['context_length']=self.hdp_ngram.tree.keys()
# 			df_concent.to_csv(self.result_path+'shared-ngram-concentration.csv', index=False)
			hdf5_store.put(
							"shared/concentration",
							df_concent,
# 							encoding='utf-8'
							)
		
			# Sublex
			df_assignment_sl = pd.DataFrame(
									self.varpar_assignment
									,
									columns=[("sublex_%i" % table_id)
												for table_id in range(self.T)
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
# 			df_stick_sl.to_csv(self.result_path+"SubLexica_stick.csv", index=False, encoding='utf-8')
			hdf5_store.put(
							"sublex/stick",
							df_stick_sl,
# 							encoding='utf-8'
							)
		
			df_concent=pd.DataFrame(
						[
							np.array((sublex.varpar_concent_shape,sublex.varpar_concent_inv_scale))
							for sublex
							in self.sublex_ngrams
						]
						+
						[self.varpar_concent]
						,
						columns=('shape','inverse_scale')
						,
						index=[('%igram_%i' % (self.n,sublex_id))
								for sublex_id
								in xrange(self.T)
								]+['sublexicalization']
						)
# 			df_concent.to_csv(self.result_path+"Sublexica_concent.csv", index=False, encoding='utf-8')
			hdf5_store.put(
							"sublex/concentration",
							df_concent,
# 							encoding='utf-8'
							)
		
			for sblx_id,sublex in enumerate(self.sublex_ngrams):
				for context,bottom_rst in sublex.restaurants.iteritems():
# 					coded_context = '_'.join([decoder[code] for code in context])#.encode('utf-8')
					coded_context = '_'.join(map(str,context))

					df_assignment = pd.DataFrame(
										bottom_rst.varpar_assignment.flatten()[:,np.newaxis]
										,
										columns=["p"]
	# 									columns=[
	# 											('cluster_id_%i' % table_id)
	# 											for table_id in range(bottom_rst.T)
	# 											]
										)
					df_assignment['customer_id']=np.repeat(
													np.arange(bottom_rst.varpar_assignment.shape[0])
													,
													bottom_rst.varpar_assignment.shape[1]
												)
					df_assignment['customer_value']=pd.Series(
														np.repeat(
															bottom_rst.customers
															,
															bottom_rst.varpar_assignment.shape[1]
															)
														)
					df_assignment['cluster_id']=np.tile(
													np.arange(bottom_rst.varpar_assignment.shape[1])
													,
													bottom_rst.varpar_assignment.shape[0]
												)
					# df_assignment = pd.DataFrame(
					# 					bottom_rst.varpar_assignment
					# 					,
					# 					columns=[("cluster_id_%i" % table_id)
					# 								for table_id in range(bottom_rst.T)]
					# 					)
					# df_assignment['customer_id']=df_assignment.index
# 					df_assignment.to_csv(self.result_path+("sublex-%i_ngram-context_%s_assignment.csv" % (sblx_id,coded_context)),
# 											index=False,
# 											encoding='utf-8'
# 											)
					hdf5_store.put(
						("sublex/sublex_%i/context_%s/assignment"
						% (sblx_id,coded_context))
						,
						df_assignment
						,
# 						encoding='utf-8'
						)
			
					df_stick=pd.DataFrame(bottom_rst.varpar_stick, columns=('beta_par1','beta_par2'))
					df_stick['cluster_id']=df_stick.index
# 					df_stick.to_csv(self.result_path+("sublex-%i_ngram-context_%s_stick.csv" % (sblx_id,coded_context)),
# 									index=False,
# 									encoding='utf-8'
# 									)
					hdf5_store.put(
						("sublex/sublex_%i/context_%s/stick"
						% (sblx_id,coded_context))
						,
						df_stick
						,
# 						encoding='utf-8'
						)
					df_label=pd.DataFrame(
									bottom_rst.varpar_label.flatten()[:,np.newaxis],
									columns=["p"]
									)
					df_label['cluster_id']=np.repeat(
												np.arange(bottom_rst.varpar_label.shape[0])
												,
												bottom_rst.varpar_label.shape[1]
											)
					df_label['value']=pd.Series(
										np.tile(
											np.arange(bottom_rst.varpar_label.shape[1])
											,
											bottom_rst.varpar_label.shape[0]
										)
										)
					# df_label=pd.DataFrame(bottom_rst.varpar_label, columns=inventory)
					# df_label['cluster_id']=df_label.index
# 					df_label.to_csv(self.result_path+("sublex-%i_ngram-context_%s_label.csv" % (sblx_id,coded_context)),
# 									index=False,
# 									encoding='utf-8'
# 									)
					hdf5_store.put(
						("sublex/sublex_%i/context_%s/label"
						% (sblx_id,coded_context))
						,
						df_label
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
# 		logger.info("Variational parameters saved.")
		
	def _update_varpar_concent(self):
# 		self.varpar_concent[0] = self.concent_priors[0]+self.T-1
		self.varpar_concent[1] = self.concent_priors[1]-np.sum(self.digamma_diffs[:,1])
		self.concent_mean=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=0)[:-1]+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[:,:0:-1],
											axis=1
												)[:,::-1],
										axis=0)+self.concent_mean
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]
		

	def _update_varpar_assignment(self):
		log_varpar_assignment = (
									np.append(
										self.digamma_diffs[:,0],
										0
										)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.digamma_diffs[:,1]
											)
										)[np.newaxis,:]
									+np.array(
										[
											word.get_E_log_likelihoods() # Output an array of length T
												for word in self.customers
										]
										)
									)
		self.varpar_assignment=np.exp(log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,np.newaxis])


	def update_varpars(self):
# 		self.hdp_ngrams.update_varpars()
# 		st = datetime.datetime.now()
		self._update_varpar_concent()
# 		print 'sublex concent', datetime.datetime.now()-st
		
# 		st = datetime.datetime.now()
		self._update_varpar_stick()
# 		print 'sublex stick', datetime.datetime.now()-st
# 		assert not np.any(np.isnan(self.varpar_stick)), ('stick\n',self.varpar_stick)
# 		new_KL=self.get_var_bound()
# 		print 'stick',new_KL-self.intermediate_var_bound
# 		self.intermediate_var_bound=new_KL
		self.hdp_ngram.update_varpars()
	
# 		st = datetime.datetime.now()
		[sublex.update_varpars() for sublex in self.sublex_ngrams]
# 		print 'base', datetime.datetime.now()-st
# 		new_KL=self.get_var_bound()
# 		print 'base',new_KL-self.intermediate_var_bound
# 		self.intermediate_var_bound=new_KL

# 		st = datetime.datetime.now()
		self._update_varpar_assignment()
# 		print 'sublex assign', datetime.datetime.now()-st
# 		assert not np.any(np.isnan(self.varpar_assignment)), ('assignment\n',self.varpar_assignment)
# 		new_KL=self.get_var_bound()
# 		print 'assign',new_KL-self.intermediate_var_bound
		
		
	def get_var_bound(self):
		"""
		Calculate the KL divergence bound based on the current variational parameters.
		We ignore the constant terms.
		"""
# 		return self.hdp_ngrams.get_var_bound()
		return (
				(
						(self.concent_mean-1)*np.sum(self.digamma_diffs[:,1]) # E[alpha-1]*E[log (1-V)]
						-
						self.varpar_concent[0]*np.log(self.varpar_concent[1])
						-
						self.concent_mean*self.concent_priors[1]
					) # E[log p(V, alpha)] - E[log q(alpha)] - constants
				+
				self.hdp_ngram.get_var_bound()
				+
				+np.sum(
					[
# 						sublex.get_sum_E_log_p_eta_lambda() # E[log p(eta)].
# 						-
# 						sublex.get_E_log_q_varpar() # E[log q(eta)]
						sublex.get_var_bound()
						for sublex in self.sublex_ngrams
					]
					) 
				+(
					np.sum(self.digamma_diffs[:,1]*np.sum(np.cumsum(
															self.varpar_assignment[...,:0:-1],
															axis=-1
															)[...,::-1], axis=0))
					+
					np.sum(self.digamma_diffs[:,0]*np.sum(self.varpar_assignment[:,:-1], axis=0))
					) # E[log p(Z | V)]
				+np.sum(
					np.array(
						[
						word.get_E_log_likelihoods()
							for word in self.customers
						]
						)# num_words x T
					*
					self.varpar_assignment
					) # E[log p(X | Z,eta)]
# 				-(
# 						(self.concent_priors[0]-1)*self.concent_log_mean
# 						-
# 						self.concent_priors[1]+self.concent_mean
# 						) # E[log q(alpha)]
				+
				(np.sum(self.digamma_diffs[:,0]*(self.varpar_stick[:,0]-1))
					+
					np.sum(self.digamma_diffs[:,1]*(self.varpar_stick[:,1]-1))
					-
					np.sum(
						sps.gammaln(self.varpar_stick),
						)
					+
					np.sum(sps.gammaln(self.sum_gamma))
					) # E[log q(V)]
				-np.sum(
					self.varpar_assignment*np.ma.log(self.varpar_assignment)
					) # E[log q(Z)]
				)
		
		
		

class SublexNgram(object):
	def __init__(self, T, n, concent_priors, id, customers, full_contexts, wrapper):
		self.id=id
		self.wrapper=wrapper
		self.n = n
		self.concent_priors=concent_priors
		self.varpar_concent_inv_scale = np.random.gamma(1,20)
		self.restaurants={context:
							DP_bottom(
									T,
									wrapper.hdp_ngram.tree[n-1][context],
									n-1,
									context,
									self
									)
							for context in full_contexts
							}
		[ngram.enter_a_restaurant(self.restaurants[ngram.context])
			for word in customers
			for ngram in word.ngrams
			]
		self.varpar_concent_shape=(T-1)*len(full_contexts)+self.concent_priors[0]
		self.concent_mean = self.varpar_concent_shape/self.varpar_concent_inv_scale
		
	def get_word_assignment_weights(self, word_ids):
		return self.wrapper.varpar_assignment[word_ids,self.id]
	
	def get_log_posterior_pred_per_word(self, test_word):
		log_pred_prob=np.float64(0)
		for ngram in test_word.ngrams:
			restaurant=None
			if ngram.context in self.restaurants.iterkeys():
				restaurant=self.restaurants[ngram.context]
			else:
				for minus_conlen,context_length in enumerate(reversed(xrange(self.n))):
					sub_context=ngram.context[minus_conlen:]
					if sub_context in self.wrapper.hdp_ngram.tree[context_length].iterkeys():
						restaurant=self.wrapper.hdp_ngram.tree[context_length][sub_context]
						break
			log_pred_prob+=spm.logsumexp(
								np.append(
									np.log(restaurant.varpar_stick[:,0])
									-
									np.log(restaurant.sum_gamma)
									,
									0
									)
								+
								np.append(
									0,
									np.cumsum(
										np.log(restaurant.varpar_stick[:,1])
										-
										np.log(restaurant.sum_gamma)
										)
									)
								+
								spm.logsumexp(
									np.log(restaurant.varpar_label) # T x |∑|
									+
									log_error_mat[ngram.target][np.newaxis,:] # 1 x |∑|
									,
									axis=1
									)
							)
		return log_pred_prob

	def _update_varpar_concent(self):
		self.varpar_concent_inv_scale = (
										self.concent_priors[1]
										-
										np.sum(
												[
												restaurant.digamma_diffs[:,1]
												for restaurant in self.restaurants.itervalues()
												]
												)
										)
		self.concent_mean = self.varpar_concent_shape/self.varpar_concent_inv_scale
	
	def set_varpar_assignment(self):
		[restaurant.set_varpar_assignment()
			for restaurant in self.restaurants.itervalues()
				]
				
	def update_varpars(self):
		self._update_varpar_concent()
		[restaurant.update_varpars()
			for restaurant in self.restaurants.itervalues()
				]
				
	def get_var_bound(self):
		return (np.sum([restaurant.get_var_bound()
					for restaurant in self.restaurants.itervalues()
					])
				-
				self.varpar_concent_shape
				*
				np.log(self.varpar_concent_inv_scale)
				-
				self.concent_priors[1]*self.concent_mean
				)

# 	def iter_bottom_restaurants(self):
# 		return self.tree[self.n-1].iteritems()
		
		
class HDPNgram(object):
	def __init__(self, T, n, concent_priors, id, full_contexts):
		self.id=id
# 		self.wrapper=wrapper
		self.n = n
		self.tree = {k:{} for k in xrange(n)}
		self.concent_priors=concent_priors
		self.varpar_concent_inv_scale = np.random.gamma(1,20,size=n)#(concent_priors[1]**-1, size=n) # The INVERSE scale of a gamma distr.
		
		self.tree[0][()]=DP_top(
							T,
							0,
							(),
							self
							)
		
# 		full_contexts = set([ngram.context for word in customers
# 						for ngram in word.ngrams
# 						])
		for context_length in xrange(1,n):
			for context in set(fc[n-1-context_length:] for fc in full_contexts):
				self.tree[context_length][context]\
								= DP(
									T,
									self.tree[context_length-1][context[1:]],
									context_length,
									context,
									self
									)
# 		for context in full_contexts:
# 			self.tree[n-1][context]\
# 								= DP_bottom(
# 									T,
# 									self.tree[n-2][context[1:]],
# 									n-1,
# 									context,
# 									self
# 									)
# 		[ngram.enter_a_restaurant(self, n)
# 			for word in customers
# 			for ngram in word.ngrams
# 			]
			
		self.varpar_concent_shape=np.array(
										[
										len(level)
										*
										(T-1)
										+
										self.concent_priors[0]
										for level in self.tree.itervalues()
										]
										)
		self.concent_mean = self.varpar_concent_shape/self.varpar_concent_inv_scale
# 	def get_word_assignment_weights(self, word_ids):
# 		return self.wrapper.varpar_assignment[word_ids,self.id]
# 		
# 		
# 	def get_E_log_q_varpar(self):
# 		return np.sum(
# 					[restaurant.get_E_log_q_varpar()
# 						for level in self.tree.itervalues()
# 							for restaurant in level.itervalues()
# 							]
# 					)
# 	
# 	def get_sum_E_log_p_eta_lambda(self):
# 		return np.sum(
# 					[restaurant.get_sum_E_log_p_varpars()
# 						for level in self.tree.itervalues()
# 							for restaurant in level.itervalues()
# 							]
# 					)
					
					
# 	def get_log_posterior_pred(self, test_data):
# 		test_word = Word(test_data, self.n, -1)
# 		log_pred_prob=0.0
# 		for ngram in test_word.ngrams:
# 			for minus_conlen,context_length in enumerate(reversed(xrange(self.n))):
# 				sub_context=ngram.context[minus_conlen:]
# 				if sub_context in self.tree[context_length].iterkeys():
# 					restaurant=self.tree[context_length][sub_context]
# 					log_mean_stick = np.log(restaurant.varpar_stick)-np.log(np.sum(restaurant.varpar_stick, axis=1))[:,np.newaxis]
# 					log_pred_prob+=spm.logsumexp(
# 										np.append(
# 											log_mean_stick[:,0],
# 											0
# 											)
# 										+
# 										np.append(
# 											0,
# 											np.cumsum(
# 												log_mean_stick[:,1]
# 												)
# 											)
# 										+
# 										spm.logsumexp(
# 											np.log(restaurant.varpar_label) # T x |∑|
# 											+
# 											log_error_mat[ngram.target][np.newaxis,:] # 1 x |∑|
# 											,
# 											axis=1
# 											)
# 									)
# 					break
# 		return log_pred_prob

	def _update_varpar_concent(self):
		self.varpar_concent_inv_scale = (
										self.concent_priors[1]
										-
										np.array(
											[
											np.sum(
												[
												restaurant.digamma_diffs[:,1]
												for restaurant in level.itervalues()
												]
												)
											for level in self.tree.itervalues()
											]
											)
										)
		self.concent_mean = self.varpar_concent_shape/self.varpar_concent_inv_scale

	def set_varpar_assignment(self):
		[restaurant.set_varpar_assignment()
			for level in self.tree.itervalues()
				for restaurant in level.itervalues()
				]
				
	def update_varpars(self):
		self._update_varpar_concent()
		[restaurant.update_varpars()
			for level in self.tree.itervalues()
				for restaurant in level.itervalues()
				]
				
	def get_var_bound(self):
		return (np.sum([restaurant.get_var_bound()
					for level in self.tree.itervalues()
						for restaurant in level.itervalues()
						])
				-
				np.sum(
					self.varpar_concent_shape
					*
					np.log(self.varpar_concent_inv_scale)
					)
				-
				self.concent_priors[1]*np.sum(self.concent_mean)
				)


	def iter_bottom_restaurants(self):
		return self.tree[self.n-1].iteritems()
		
		
		
class DP(object):
	def __init__(self, T, mother, context_length, context, hdp):
		self.mother = mother
		self.hdp=hdp
		self.context_length=context_length
		self.context = context
# 		self.concent_priors = concent_priors
# 		self.varpar_concent=np.zeros(2)
# 		self.varpar_concent[0] = self.concent_priors[0]+T-1 # The shape of a gamma distr.
# 		self.varpar_concent[1] = np.random.gamma(self.concent_priors[1]**-1) # The INVERSE scale of a gamma distr.
# 		self.hdp.concent_mean[self.context_length]=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(T-1,2)
												) # gamma in Blei and Jordan.
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]
		self.varpar_label = np.random.dirichlet(np.ones(num_symbols), T)
		self.tau_x_error = np.matmul(
								self.varpar_label,
								log_error_mat
								)
		self.T = T
		self.children = []
		self.new_id = 0
		self.id = self.mother.add_customer(self)
		
		
	def add_customer(self, child):
		self.children.append(child)
		issued_id = self.new_id
		self.new_id+=1
		return issued_id
		
	def set_varpar_assignment(self):
		self.num_children = len(self.children)
		self.num_tables_child = self.children[0].varpar_label.shape[0] # T for children.
		self.varpar_assignment = np.random.dirichlet(
											np.ones(self.T),
											(self.num_children, self.num_tables_child)
											) # phi in Blei and Jordan.
		assert self.varpar_assignment.size, 'Empty restaurant created.'
		self.phi_x_tau_x_error=np.matmul(self.varpar_assignment.reshape(self.num_children*self.num_tables_child, self.T),
										self.tau_x_error
										).reshape(
														self.num_children,
														self.num_tables_child,
														num_symbols
														)
		self.var_bound = -np.inf
		
# 	def _update_varpar_concent(self):
# # 		self.varpar_concent[0] = self.concent_priors[0]+self.T-1
# 		self.varpar_concent[1] = self.concent_priors[1]-np.sum(self.digamma_diffs[:,1])
# 		self.hdp.concent_mean[self.context_length]=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=(0, 1))[:-1]+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[...,:0:-1],
											axis=-1
												)[...,::-1],
										axis=(0,1)
										)+self.hdp.concent_mean[self.context_length]
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]



	def _update_varpar_assignment(self, children_label_np):
		log_varpar_assignment = (
									np.append(
										self.digamma_diffs[:,0],
										0
										)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.digamma_diffs[:,1]
											)
										)[np.newaxis,:]
									+np.matmul(
										children_label_np,
										np.transpose(self.tau_x_error) # |∑|xT
										)
								)
		varpar_assignment_2dim=np.exp(log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,np.newaxis])
		self.varpar_assignment=varpar_assignment_2dim.reshape(self.num_children, self.num_tables_child, self.T)
		self.phi_x_tau_x_error=np.matmul(
									varpar_assignment_2dim,
									self.tau_x_error).reshape(
														self.num_children,
														self.num_tables_child,
														num_symbols
														)


	def _update_varpar_label(self, children_label_np):
		self.log_varpar_label = (
								self.mother.phi_x_tau_x_error[self.id]
								+np.matmul(
									np.matmul(
										np.transpose(self.varpar_assignment.reshape(
																	self.num_children*self.num_tables_child,
																	self.T
																	)
													), # Tx(num_children*num_tables_child)
										children_label_np # (num_children*num_tables_child)x|∑|
										),
									log_error_mat # |∑|x|∑|
									)
								)
		self.varpar_label=np.exp(self.log_varpar_label-spm.logsumexp(self.log_varpar_label, axis=-1)[:, np.newaxis])
		self.tau_x_error = np.matmul(
									self.varpar_label,
									log_error_mat
									)
		assert not np.any(np.isnan(self.tau_x_error)), ('tau_x_error\n',self.tau_x_error)
									
	def update_varpars(self):
		children_label_np = np.array([child.varpar_label for child in self.children]).reshape(
																self.num_children*self.num_tables_child,
																num_symbols
																)# (num_children*num_tables_child)x|∑|
# 		self._update_varpar_concent()
		self._update_varpar_stick()
		self._update_varpar_label(children_label_np)
		self._update_varpar_assignment(children_label_np)
			
	
	def get_var_bound(self):
		return self.get_sum_E_log_p_varpars()-self.get_E_log_q_varpar()

	def get_sum_E_log_p_varpars(self):
		return (
					(
# 						self.varpar_concent[0]*self.concent_log_mean
# 						-
# 						self.hdp.concent_mean[self.context_length]*self.varpar_concent[1]
# 						+
						(self.hdp.concent_mean[self.context_length]-1)*np.sum(self.digamma_diffs[:,1]) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+
					np.sum(
									self.mother.phi_x_tau_x_error[self.id]
									*self.varpar_label # Tx|∑|
								) # E[log p(eta_m | Z_m-1, eta_m-1)]
					+
					np.sum(
							self.digamma_diffs[:,1]*np.sum(np.cumsum(
															self.varpar_assignment[...,:0:-1],
															axis=-1
															)[...,::-1], axis=(0,1))
							+
							self.digamma_diffs[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
							) # E[log p(Z | V)]
					)

				
	def get_E_log_q_varpar(self):
		return(
# 					(
# 						(self.concent_priors[0]-1)*self.concent_log_mean
# 						-
# 						self.concent_priors[1]+self.hdp.concent_mean[self.context_length]
# 					) # E[log q(alpha)]
# 					+
					(np.sum(self.digamma_diffs[:,0]*(self.varpar_stick[:,0]-1))
						+
						np.sum(self.digamma_diffs[:,1]*(self.varpar_stick[:,1]-1))
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							)
						+
						np.sum(sps.gammaln(self.sum_gamma))
					) # E[log q(V)]
					+
					np.sum(self.varpar_label*np.ma.log(self.varpar_label)) # E[log q(Z)]
					+
					np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment)) # E[log q(eta)]
					)
		
class DP_bottom(object):
	def __init__(self, T, mother, context_length, context, sublex):
		self.sublex=sublex
		self.mother = mother
		self.context_length=context_length
		self.context = context
# 		self.concent_priors = concent_priors
# 		self.varpar_concent=np.zeros(2)
# 		self.varpar_concent[0] = self.concent_priors[0]+T-1 # The shape of a gamma distr.
# 		self.varpar_concent[1] = np.random.gamma(self.concent_priors[1]**-1) # The INVERSE scale of a gamma distr.
# 		self.sublex.concent_mean[self.context_length]=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(T-1,2)
												) # gamma in Blei and Jordan.
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]
		self.varpar_label = np.random.dirichlet(np.ones(num_symbols), T) # Emission probability approximated by a multinomial distribution.
		self.T = T
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
		self.varpar_assignment = np.random.dirichlet(np.ones(self.T), len(self.customers)) # phi in Blei and Jordan.
		assert self.varpar_assignment.size, 'Empty restaurant created.'
		self.phi_x_tau_x_error = np.matmul(
								self.varpar_assignment,
								np.matmul(
									self.varpar_label,
									log_error_mat
									)
								)
								
# 	def _update_varpar_concent(self):
# # 		self.varpar_concent[0] = self.concent_priors[0]+self.T-1
# 		self.varpar_concent[1] = self.concent_priors[1]-np.sum(self.digamma_diffs[:,1])
# 		self.sublex.concent_mean[self.context_length]=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=0)[:-1]+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[:,:0:-1],
											axis=1
												)[:,::-1],
										axis=0)+self.sublex.concent_mean
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]
		
	def _update_varpar_assignment(self):
		log_varpar_assignment = (
									np.append(
										self.digamma_diffs[:,0],
										0
										)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.digamma_diffs[:,1]
											)
										)[np.newaxis,:]
									+(np.matmul(
										log_error_mat[self.customers], # Nx|∑|
										np.transpose(self.varpar_label) # |∑|xT
										)
										*
										self.sublex.get_word_assignment_weights(self.word_ids)[:,np.newaxis]
										)
								)
		self.varpar_assignment=np.exp(
									log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,np.newaxis]
									)
		self.phi_x_tau_x_error = np.matmul(
									self.varpar_assignment,
									np.matmul(
										self.varpar_label,
										log_error_mat
										)
									)
		assert not np.any(np.isnan(self.phi_x_tau_x_error)), ('phi_x_tau_x_error\n',self.phi_x_tau_x_error)
		
		
									
											
	def _update_varpar_label(self):
		log_varpar_label = (
								self.mother.phi_x_tau_x_error[self.id]
								+
								np.matmul(
									np.transpose(self.varpar_assignment), # TxN
									self.sublex.get_word_assignment_weights(self.word_ids)[:,np.newaxis] #やや怪しい。
									*
									log_error_mat[self.customers] # Nx|∑|
									) #Tx|∑|
								)
		self.varpar_label=np.exp(log_varpar_label-spm.logsumexp(log_varpar_label, axis=-1)[:, np.newaxis])
								
	def update_varpars(self):
# 		self._update_varpar_concent()
		self._update_varpar_stick()
		assert not np.any(np.isnan(self.varpar_stick)), ('stick\n',self.varpar_stick)
		self._update_varpar_label()
		assert not np.any(np.isnan(self.varpar_label)), ('label\n',self.varpar_label)
		self._update_varpar_assignment()
		assert not np.any(np.isnan(self.varpar_assignment)), ('assignment\n',self.varpar_assignment)
			
			
	def get_var_bound(self):
		return (
						self.get_sum_E_log_p_varpars()
						-self.get_E_log_q_varpar()
# 						+np.sum(
# 							[self.phi_x_tau_x_error[customer_id,customer_value]
# 								for customer_id,customer_value
# 								in enumerate(self.customers)
# 								]
# 								)
						)
		
	def get_sum_E_log_p_varpars(self):
# 		print self.id
		return (
					(
# 						self.varpar_concent[0]*self.concent_log_mean
# 						-
# 						self.sublex.concent_mean[self.context_length]*self.varpar_concent[1]
# 						+
						(self.sublex.concent_mean-1)*np.sum(self.digamma_diffs[:,1]) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+
					np.sum(
									self.mother.phi_x_tau_x_error[self.id]
									*self.varpar_label # Tx|∑|
								) # E[log p(eta_m | Z_m-1, eta_m-1)]
					+
					np.sum(
							self.digamma_diffs[:,1]*np.sum(np.cumsum(
															self.varpar_assignment[...,:0:-1],
															axis=-1
															)[...,::-1], axis=0)
							+
							self.digamma_diffs[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=0)
							) # E[log p(Z | V)]
					)
	
		
				
	def get_E_log_q_varpar(self):
		return(
# 					(
# 						(self.concent_priors[0]-1)*self.concent_log_mean
# 						-
# 						self.concent_priors[1]+self.sublex.concent_mean[self.context_length]
# 					) # E[log q(alpha)]
# 					+
					(np.sum(self.digamma_diffs[:,0]*(self.varpar_stick[:,0]-1))
						+
						np.sum(self.digamma_diffs[:,1]*(self.varpar_stick[:,1]-1))
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							)
						+
						np.sum(sps.gammaln(self.sum_gamma))
					) # E[log q(V)]
					+
					np.sum(self.varpar_label*np.ma.log(self.varpar_label)) # E[log q(Z)]
					+
					np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment)) # E[log q(eta)]
					)
	
class DP_top(object):
	def __init__(self, T, context_length, context, hdp):
		self.hdp=hdp
		self.context_length=context_length
		self.context = context
# 		self.concent_priors = concent_priors
# 		self.varpar_concent=np.zeros(2)
# 		self.varpar_concent[0] = self.concent_priors[0]+T-1 # The shape of a gamma distr.
# 		self.varpar_concent[1] = np.random.gamma(self.concent_priors[1]**-1) # The INVERSE scale of a gamma distr.
# 		self.hdp.concent_mean[self.context_length]=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		self.varpar_stick = np.random.gamma(1,#10,
												20,#0.1,
												size=(T-1,2)
												) # gamma in Blei and Jordan.
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]
		self.varpar_label = np.random.dirichlet(np.ones(num_symbols), T)
		self.tau_x_error = np.matmul(
								self.varpar_label,
								log_error_mat
								)
		self.T = T
		self.new_id=0
		self.children=[]
		
		
	def add_customer(self, child):
		self.children.append(child)
		issued_id = self.new_id
		self.new_id+=1
		return issued_id
		
	def set_varpar_assignment(self):
		self.num_children = len(self.children)
		self.num_tables_child = self.children[0].varpar_label.shape[0] # T for children.
		self.varpar_assignment = np.random.dirichlet(
											np.ones(self.T),
											(self.num_children, self.num_tables_child)
											) # phi in Blei and Jordan.
		self.phi_x_tau_x_error=np.matmul(self.varpar_assignment.reshape(self.num_children*self.num_tables_child, self.T),
										self.tau_x_error
										).reshape(
														self.num_children,
														self.num_tables_child,
														num_symbols
														)
		
# 	def _update_varpar_concent(self):
# # 		self.varpar_concent[0] = self.concent_priors[0]+self.T-1
# 		self.varpar_concent[1] = self.concent_priors[1]-np.sum(self.digamma_diffs[:,1])
# 		self.hdp.concent_mean[self.context_length]=self.varpar_concent[0]/self.varpar_concent[1]
# 		self.concent_log_mean=sps.digamma(self.varpar_concent[0])-np.log(self.varpar_concent[1])
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=(0, 1))[:-1]+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[...,:0:-1],
											axis=-1
												)[...,::-1],
										axis=(0,1)
										)+self.hdp.concent_mean[self.context_length]
		self.sum_gamma = np.sum(self.varpar_stick, axis=-1)
		self.digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(self.sum_gamma)[:, np.newaxis]



	def _update_varpar_assignment(self, children_label_np):
		log_varpar_assignment = (
									np.append(
# 										sps.digamma(self.varpar_stick[:,0])-digamma_gamma1plus2,
										self.digamma_diffs[:,0],
										0
										)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											self.digamma_diffs[:,1]
											)
										)[np.newaxis,:]
									+np.matmul(
										children_label_np,
										np.transpose(self.tau_x_error)
										)
								)
		varpar_assignment_2dim=np.exp(log_varpar_assignment-spm.logsumexp(log_varpar_assignment, axis=-1)[:,np.newaxis])
		self.varpar_assignment=varpar_assignment_2dim.reshape(self.num_children, self.num_tables_child, self.T)
		self.phi_x_tau_x_error=np.matmul(
									varpar_assignment_2dim,
									self.tau_x_error).reshape(
														self.num_children,
														self.num_tables_child,
														num_symbols
														)
											
	def _update_varpar_label(self, children_label_np):
		log_varpar_label = (
								np.matmul(
									np.matmul(
										np.transpose(self.varpar_assignment.reshape(
																	self.num_children*self.num_tables_child,
																	self.T
																	)
													), # Tx(num_children*num_tables_child)
										children_label_np # (num_children*num_tables_child)x|∑|
										),
									log_error_mat # |∑|x|∑|
									)
								)
		self.varpar_label=np.exp(log_varpar_label-spm.logsumexp(log_varpar_label, axis=-1)[:, np.newaxis])
		self.tau_x_error = np.matmul(
								self.varpar_label,
								log_error_mat
								)
									
	def update_varpars(self):
		children_label_np = np.array([child.varpar_label for child in self.children]).reshape(
																self.num_children*self.num_tables_child,
																num_symbols
																)# (num_children*num_tables_child)x|∑|
# 		self._update_varpar_concent()
		self._update_varpar_stick()
		self._update_varpar_label(children_label_np)
		self._update_varpar_assignment(children_label_np)
		
	def get_var_bound(self):
		return self.get_sum_E_log_p_varpars()-self.get_E_log_q_varpar()
		
	def get_sum_E_log_p_varpars(self):
		return (
					(
# 						self.varpar_concent[0]*self.concent_log_mean
# 						-
# 						self.hdp.concent_mean[self.context_length]*self.varpar_concent[1]
# 						+
						(self.hdp.concent_mean[self.context_length]-1)*np.sum(self.digamma_diffs[:,1]) # E[alpha-1]*E[log (1-V)]
					) # E[log p(V, alpha)] joint distr is easier to compute than likelihood and prior independently.
					+ # E[log p(eta_m | Z_m-1, eta_m-1)] is a constant and is skipped.
					np.sum(
							self.digamma_diffs[:,1]*np.sum(np.cumsum(
																self.varpar_assignment[...,:0:-1],
																axis=-1
																)[...,::-1], axis=(0,1))
							+
							self.digamma_diffs[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
							) # E[log p(Z | V)]
				)

	def get_E_log_q_varpar(self):
		return(
# 					(
# 						(self.concent_priors[0]-1)*self.concent_log_mean
# 						-
# 						self.concent_priors[1]+self.hdp.concent_mean[self.context_length]
# 					) # E[log q(alpha)]
# 					+
					(np.sum(self.digamma_diffs[:,0]*(self.varpar_stick[:,0]-1))
						+
						np.sum(self.digamma_diffs[:,1]*(self.varpar_stick[:,1]-1))
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							)
						+
						np.sum(sps.gammaln(self.sum_gamma))
					) # E[log q(V)]
					+
					np.sum(self.varpar_label*np.ma.log(self.varpar_label)) # E[log q(Z)]
					+
					np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment)) # E[log q(eta)]
					)
	
class Word(object):
	def __init__(self, string, n, id):
		self.id=id
		self.ngrams = [Ngram(window,self)
						for window in zip(*[([0]*(n-1)+string+[1])[i:] for i in range(n)])
						]
		
		
# 	def get_sum_E_log_p_x_Z(self):
# 		return np.sum(
# 					[
# 						ngram.get_E_log_likelihoods() for ngram in self.ngrams
# 					]
# 					)
					
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
		self.restaurants = []
		self.ids = []
		self.context = window[:-1]
		self.target = window[-1]
		
	def enter_a_restaurant(self, restaurant):
# 		print self.context,hdp_ngram.tree[n-1]
# 		restaurant = hdp_ngram.tree[n-1][self.context]
		self.ids.append(restaurant.add_customer(self.target, self.word.id))
		self.restaurants.append(restaurant)
		
	
		
# 	def get_E_log_likelihoods(self):
# 		return np.sum(
# 							[
# 							r.phi_x_tau_x_error[i, self.target]
# 							for i,r in zip(self.ids, self.restaurants)
# 							]
# 						)
						
	def get_E_log_likelihoods(self):
		return np.array(
						[
							r.phi_x_tau_x_error[i, self.target]
							for i,r in zip(self.ids, self.restaurants)
						]
					)

# def code_data(str_data):
# 	inventory = sorted(set(reduce(lambda x,y: x+y, str_data)))
# 	num_symbols = len(inventory)
# 	encoder = {symbol:code for code,symbol in enumerate(inventory, start=2)}
# 	decoder = {code:symbol for code,symbol in enumerate(inventory, start=2)}
# 	decoder[0]='$' # Special code reserved for initial symbol.
# 	decoder[1]='#' # Special code reserved for end symbol.
# 	coded_data = [map(lambda s: encoder[s],phrase) for phrase in str_data]
# # 	print coded_data
# 	return (coded_data,encoder,decoder)

# def code_data(series):
# 	str_data=[word.split(',') for index, word in series.iteritems()]
# 	inventory = list(set(itertools.chain.from_iterable(str_data)))
# # 	print inventory
# 	num_symbols = len(inventory)
# 	encoder = {symbol:code for code,symbol in enumerate(inventory, start=2)}
# 	decoder = {code:symbol for symbol,code in encoder.iteritems()}
# 	decoder[0]=u'$' # Special code reserved for initial symbol.
# 	decoder[1]=u'#' # Special code reserved for end symbol.
# 	coded_data = [map(lambda s: encoder[s],phrase) for phrase in str_data]
# # 	print coded_data
# 	return (coded_data,decoder)

def code_data(training_data,test_data=None):
	str_training_data=[word.split(',') for index, word in training_data.iteritems()]
	if test_data is None:
		str_data=str_training_data
	else:
		str_test_data=[word.split(',') for index, word in test_data.iteritems()]
		str_data=str_training_data+str_test_data
	inventory = list(set(itertools.chain.from_iterable(str_data)))

	num_symbols = len(inventory)

	encoder = {symbol:code for code,symbol in enumerate(inventory, start=2)}
	decoder = {code:symbol for code,symbol in enumerate(inventory, start=2)}

	decoder[0]=u'START' # Special code reserved for initial symbol.
	decoder[1]=u'END' # Special code reserved for end symbol.

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
	result_path = ('/om/user/tmorita/results_shared_base/'+os.path.splitext(datapath.split('/')[-1])[0]+'_'+now+'/')
# 	result_path = ('./results/'+os.path.splitext(datapath.split('/')[-1])[0]+'_'+now+'/')
	os.makedirs(result_path)
	sl_T = int(sys.argv[2])
	n = int(sys.argv[3])
	T_base = len(decoder)*2 # Number of symbols x 2
	concent_priors = np.array((10.0,10.0)) # Gamma parameters (shape, INVERSE of scale) for prior on concentration.
# 	noise = np.float64(sys.argv[4])
	max_iters = int(sys.argv[4])
	min_increase = np.float64(sys.argv[5])
	start = datetime.datetime.now()
	vi = VariationalInference(
			sl_T,
			customers,
# 			sl_concent,
			n,
			T_base,
			concent_priors,
# 			noise,
			result_path
			)
	vi.train(max_iters, min_increase)
	vi.save_results(decoder)
	print 'Time spent',str(datetime.datetime.now()-start)
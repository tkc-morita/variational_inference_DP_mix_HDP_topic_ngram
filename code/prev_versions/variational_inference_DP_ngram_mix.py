# coding: utf-8

import numpy as np
import scipy.special as sps
import itertools,sys,datetime,os
from logging import getLogger,FileHandler,DEBUG,Formatter
import pandas as pd
import warnings

class VariationalInferenceDP(object):
	def __init__(
			self,
			T,
			customers,
			concent,
			n,
			T_base,
			base_concent,
			noise,
			result_path
			):
		self.setup_logger(result_path)
		global num_symbols
		num_symbols = len(set(reduce(lambda x,y: x+y, customers)))+2
		self.logger.info('# of symbols: %i' % num_symbols)
		
		global error_mat
		error_mat = np.full((num_symbols,)*2, noise/(num_symbols-1))
		np.fill_diagonal(
						error_mat,
						1-noise
						)
		error_mat = np.log(error_mat)
		self.logger.info('Noise of transmission from tables to customers: %f' % base_concent)
# 		print error_mat
		
		self.customers = [Word(word, n, self) for word in customers]
		num_customers = len(customers)
		self.logger.info('# of words: %i' % num_customers)
		self.varpar_assignment = np.random.dirichlet(np.ones(T), num_customers) # phi in Blei and Jordan.
		
		self.varpar_stick = np.random.gamma(10, 0.1, size=(T-1,2)) # gamma in Blei and Jordan.
		self.T = T
		self.logger.info('(max) # of tables: %i' % T)
		
		
		self.concent = concent
		self.logger.info('Concentration for sublexicalization: %f' % concent)
		
		self.base_distrs = [HDPNgram(T_base, n, base_concent) for table_id in xrange(T)]
		for bd in self.base_distrs:
			[ngram.enter_a_restaurant(bd, n)
				for word in self.customers
				for ngram in word.ngrams
				]
			bd.set_varpar_assignment()
		
		self.logger.info('Concentration for base distribution: %f' % base_concent)

	
		
						
		self.result_path = result_path 
		self.logger.info('Initialization complete.')


		
	def setup_logger(self,result_path):
		self.logger = getLogger(__name__)
		date = datetime.datetime.today() #Check the date and time of the experiment start.
		handler = FileHandler(filename=result_path+'VI_DP_ngram.log')	#Define the handler.
		handler.setLevel(DEBUG)
		formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')	#Define the log format.
		handler.setFormatter(formatter)
		self.logger.setLevel(DEBUG)
		self.logger.addHandler(handler)	#Register the handler for the logger.
		self.logger.info("Logger set up.")
		
	
	def run(self, max_iters, min_increase):
		self.logger.info("Main loop started.")
		self.logger.info("Max iteration is %i." % max_iters)
		self.logger.info("Will be terminated if KL bound is improved less than %f." % min_increase)
		converged=False
		iter_id=0
		self.current_KL_bound = -np.inf
		while iter_id<max_iters:
			iter_id+=1
			self.update_varpars()
			self.logger.info("Variational parameters updated. Iteration ID: %i" % iter_id)
			new_KL_bound = self.get_KL_bound()
			improvement = new_KL_bound-self.current_KL_bound
			if np.isnan(new_KL_bound):
				raise Exception("nan detected.")
			if improvement<=0:
				self.logger.error("KL bound decreased. Something wrong.")
				raise Exception("KL bound decreased. Something wrong.")
			elif improvement<min_increase:
				converged = True
				break
			else:
				self.current_KL_bound=new_KL_bound
		if converged:
			self.logger.info('Converged after %i iterations.' % iter_id)
		else:
			self.logger.error('Failed to converge after max iterations.')
		self.logger.info('Final KL bound is %f.' % self.current_KL_bound)
		

		
	def save_results(self):
		np.savetxt(self.result_path+"sublexicalization_stick.csv", self.varpar_stick, delimiter=",")
		np.savetxt(self.result_path+"sublexicalization_assignment.csv", self.varpar_assignment, delimiter=",")
		self.logger.info("Variational parameters saved.")
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=0)[:-1]#+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[:,::-1],
											axis=1
												)[:,::-1],
										axis=0)[1:]+(self.concent-1)
# 		print self.T
# 		print self.varpar_stick
		
		
	def _update_varpar_assignment(self):
		digamma_gamma1plus2 = sps.digamma(np.sum(self.varpar_stick, axis=1))
# 		print np.array(
# 										[
# 											word.get_E_log_likelihoods() # Output an array of length T
# 												for word in self.customers
# 										]
# 										)
# 		print 'update assignment'
# 		print 'digamma_gamma1plus2', digamma_gamma1plus2
# 		print '
# 		print np.append(sps.digamma(self.varpar_stick[:,0]), 0)-digamma_gamma1plus2
# 		print np.append(
# 										0,
# 										np.cumsum(
# 											sps.digamma(self.varpar_stick[:,1])
# 											-digamma_gamma1plus2
# 											)
# 										)
# 		print np.array(
# 										[
# 											word.get_E_log_likelihoods() # Output an array of length T
# 												for word in self.customers
# 										]
# 										)
		self.varpar_assignment = np.exp(
									np.append(sps.digamma(self.varpar_stick[:,0])-digamma_gamma1plus2, 0)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											sps.digamma(self.varpar_stick[:,1])
											-digamma_gamma1plus2
											)
										)[np.newaxis,:]
									+np.array(
										[
											word.get_E_log_likelihoods() # Output an array of length T
												for word in self.customers
										]
										)
									)
		self.varpar_assignment/=np.sum(self.varpar_assignment, axis=-1)[:,np.newaxis]
# 		print self.varpar_assignment


	def update_varpars(self):
		self._update_varpar_stick()
		print 'stick',self.get_KL_bound()
		[table.update_varpars() for table in self.base_distrs]
		print 'base',self.get_KL_bound()
		self._update_varpar_assignment()
		print 'assign',self.get_KL_bound()
		
		
	def get_KL_bound(self):
		"""
		Calculate the KL divergence bound based on the current variational parameters.
		We ignore the constant terms.
		"""
		sum_gamma = np.sum(self.varpar_stick, axis=-1)
		digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
		cumsum_phis = np.cumsum(
							self.varpar_assignment[:,::-1],
							axis=-1
							)[:,::-1]
		print 'KL'
		print np.sum(
# 					digamma_diffs[:,0]
# 					+
					digamma_diffs[:,1]*(self.concent-1)
					) # E[log p(V | alpha)]
		print np.sum(
					[
						table.get_sum_E_log_p_eta_lambda() # E[log p(eta)].
						for table in self.base_distrs
					]
					)
		print -np.sum(
					[
						table.get_E_log_varpar() # E[log q(eta)]
						for table in self.base_distrs
					]
					) 
		print np.sum(
					digamma_diffs[:,1]*np.sum(cumsum_phis[:,1:], axis=0) #cumsumの範囲がおかしい
					+
					digamma_diffs[:,0]*np.sum(self.varpar_assignment[:,:-1], axis=0)
					) # E[log p(Z | V)]
		print -np.sum(
					digamma_diffs[:,0]*(self.varpar_stick[:,0]-1)
					+
					digamma_diffs[:,1]*(self.varpar_stick[:,1]-1)
					-
					np.sum(
						sps.gammaln(self.varpar_stick),
						axis=1
						)
					+
					sps.gammaln(sum_gamma)
					) # E[log q(V)]
		print -np.sum(
					self.varpar_assignment*np.ma.log(self.varpar_assignment)
					) # E[log q(Z)]
		print np.sum(
					np.array(
						[
						word.get_E_log_likelihoods()
							for word in self.customers
						]# num_words x T
						)
					*
					self.varpar_assignment
					) # E[log p(X | Z,eta)]
		return (
				np.sum(
# 					digamma_diffs[:,0]
# 					+
					digamma_diffs[:,1]*(self.concent-1)
					) # E[log p(V | alpha)]
				+np.sum(
					[
						table.get_sum_E_log_p_eta_lambda() # E[log p(eta)].
# 						+
# 						table.get_E_log_likelihoods() # E[log p(X | Z, eta)]
						-
						table.get_E_log_varpar() # E[log q(eta)]
						for table in self.base_distrs
					]
					) 
				+np.sum(
					digamma_diffs[:,1]*np.sum(cumsum_phis[:,1:], axis=0)
					+
					digamma_diffs[:,0]*np.sum(self.varpar_assignment[:,:-1], axis=0)
					) # E[log p(Z | V)]
				-np.sum(
					digamma_diffs[:,0]*(self.varpar_stick[:,0]-1)
					+
					digamma_diffs[:,1]*(self.varpar_stick[:,1]-1)
					-
					np.sum(
						sps.gammaln(self.varpar_stick),
						axis=1
						)
					+
					sps.gammaln(sum_gamma)
					) # E[log q(V)]
				-np.sum(
					self.varpar_assignment*np.ma.log(self.varpar_assignment)
					) # E[log q(Z)]
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
				)
		


		
class HDPNgram(object):
	def __init__(self, T, n, concent):
		self.n = n
		self.tree = {k:{} for k in xrange(n)}
		self.tree[0][()]=DP_top(
							T,
							concent,
							)
		for context_length in xrange(1,n):
			for context in itertools.product(range(num_symbols),repeat=context_length):
				for initial_fill_length in xrange(n-context_length):
					initial_and_context = (num_symbols,)*initial_fill_length+context
					if context_length+initial_fill_length==n-1:
						self.tree[context_length+initial_fill_length][initial_and_context]\
								= DP_bottom(
									T,
									concent,
									self.tree[context_length+initial_fill_length-1][initial_and_context[1:]]
									)
					else:
						self.tree[context_length+initial_fill_length][initial_and_context]\
								= DP(
									T,
									concent,
									self.tree[context_length+initial_fill_length-1][initial_and_context[1:]]
									)

		
		
	def get_E_log_varpar(self):
		return np.sum(
					[restaurant.get_E_log_varpar()
						for level in self.tree.itervalues()
							for restaurant in level.itervalues()
							]
					)
	
	def get_sum_E_log_p_eta_lambda(self):
		return np.sum(
					[restaurant.get_sum_E_log_p_parameters()
						for level in self.tree.itervalues()
							for restaurant in level.itervalues()
							]
					)
					
	def set_varpar_assignment(self):
		[restaurant.set_varpar_assignment()
			for level in self.tree.itervalues()
				for restaurant in level.itervalues()
				]
				
	def update_varpars(self):
		[restaurant.update_varpars()
			for level in self.tree.itervalues()
				for restaurant in level.itervalues()
				]


# 	def get_E_log_likelihoods(self): # Return an array of length num_words.
# 		return np.array(
# 				[
# 					word.get_E_log_likelihoods()
# 					for word in self.training_data
# 				]
# 				)
		
class DP(object):
	def __init__(self, T, concent, mother):
		self.mother = mother
		self.varpar_stick = np.random.gamma(10, 0.1, size=(T-1,2)) # gamma in Blei and Jordan.
# 								for i in xrange(n)]
		self.varpar_label = np.random.dirichlet(np.ones(num_symbols), T)
# 								[np.random.dirichlet(np.ones(num_symbols), [T]+[num_symbols]*(i))
# 								for i in xrange(n)
# 								]# Emission probability approximated by a multinomial distribution.

		self.concent = concent
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
# 		self.children = children # Represented by a list of children restaurants.
# 		self.children_label = [child.varpar_label for child in self.children]
		self.num_children = len(self.children)
		self.num_tables_child = self.children[0].varpar_label.shape[0] # T for children.
		self.varpar_assignment = np.random.dirichlet(
											np.ones(self.T),
											(self.num_children, self.num_tables_child)
											) # phi in Blei and Jordan.
		self.tau_x_error = np.matmul(
								self.varpar_label,
								error_mat
								)
		if not self.varpar_assignment.size:
			self.varpar_label=np.zeros(self.varpar_label.shape)
		
	def _update_varpar_stick(self):
# 		print np.sum(self.varpar_assignment, axis=(0, 1))[:-1]
# 		print np.sum(self.varpar_assignment, axis=(0, 1))
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=(0, 1))[:-1]#+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[...,::-1],
											axis=-1
												)[...,::-1],
										axis=(0,1)
										)[1:]+(self.concent-1)

# 		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=0)[:-1]#+1
# 		self.varpar_stick[...,1] = np.sum(
# 										np.cumsum(
# 											self.varpar_assignment[:,::-1],
# 											axis=1
# 												)[:,::-1],
# 										axis=0)[1:]+(self.concent-1)


	def _update_varpar_assignment(self, children_label_np):
		digamma_gamma1plus2 = sps.digamma(np.sum(self.varpar_stick, axis=1))
		self.varpar_assignment = np.exp(
									np.append(sps.digamma(self.varpar_stick[:,0])-digamma_gamma1plus2, 0)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											sps.digamma(self.varpar_stick[:,1])
											-digamma_gamma1plus2
											)
										)[np.newaxis,:]
									+np.matmul(
										children_label_np,
										np.transpose(self.tau_x_error) # |∑|xT
										)
								)
		denom=np.sum(self.varpar_assignment, axis=-1)[:,np.newaxis]
		self.varpar_assignment /= denom+(denom==0).astype(np.float64)
		self.varpar_assignment = self.varpar_assignment.reshape(self.num_children, self.num_tables_child, self.T)
		
									
											
	def _update_varpar_label(self, children_label_np):
		self.varpar_label = np.exp(
								np.matmul(
									self.mother.varpar_assignment[self.id], # Txmother_T
									np.matmul(
										self.mother.varpar_label, # mother_T x |∑|
										error_mat # |∑|x|∑|
										)
									)
								+np.matmul(
									np.matmul(
										np.transpose(self.varpar_assignment.reshape(
																	self.num_children*self.num_tables_child,
																	self.T
																	)
													), # Tx(num_children*num_tables_child)
										children_label_np # (num_children*num_tables_child)x|∑|
										),
									error_mat # |∑|x|∑|
									)
								)
		denom=np.sum(self.varpar_label, axis=-1)[:, np.newaxis]
		self.varpar_label /= denom+(denom==0).astype(np.float64)
		
									
	def update_varpars(self):
		if self.varpar_assignment.size:
			children_label_np = np.array([child.varpar_label for child in self.children]).reshape(
																	self.num_children*self.num_tables_child,
																	num_symbols
																	)# (num_children*num_tables_child)x|∑|
			self._update_varpar_stick()
			assert not np.sum(np.isnan(self.varpar_stick)), ('stick\n',self.varpar_stick)
			self._update_varpar_label(children_label_np)
			assert not np.sum(np.isnan(self.varpar_label)), ('label\n',self.varpar_label)
			self._update_varpar_assignment(children_label_np)
			assert not np.sum(np.isnan(self.varpar_assignment)), ('assignment\n',self.varpar_assignment)
			self.tau_x_error = np.matmul(
									self.varpar_label,
									error_mat
									)
			assert not np.sum(np.isnan(self.tau_x_error)), ('tau_x_error\n',self.tau_x_error)
	
		
	def get_sum_E_log_p_parameters(self):
		if self.varpar_assignment.size:
			sum_gamma = np.sum(self.varpar_stick, axis=-1)
			digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
			cumsum_phis = np.cumsum(
								self.varpar_assignment[...,::-1],
								axis=-1
								)[...,::-1]
# 			print 'cumsum_phis', cumsum_phis 
			output= (
					np.sum(
	# 					digamma_diffs[:,0]
	# 					+
						digamma_diffs[:,1]*(self.concent-1)
						) # E[log p(V | alpha)]
					+np.sum(
						np.matmul(
							np.matmul(
								self.mother.varpar_assignment[self.id],
								self.mother.tau_x_error
								),
							np.transpose(self.varpar_label)
							)
						) # E[log p(eta_m | Z_m-1, eta_m-1)]
					+np.sum(
						digamma_diffs[:,1]*np.sum(cumsum_phis[...,1:], axis=(0,1))
						+
						digamma_diffs[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
						) # E[log p(Z | V)]
	# 				+np.sum(
	# 					digamma_diffs[:,0]*self.varpar_stick[:,0]
	# 					+
	# 					digamma_diffs[:,1]*self.varpar_stick[:,1]
	# 					-
	# 					np.sum(
	# 						sps.gammaln(self.varpar_stick),
	# 						axis=0
	# 						)
	# 					+
	# 					sps.gammaln(sum_gamma)
	# 					) # E[log q(V)]
	# 				+np.sum(
	# 					self.varpar_assignment*np.ma.log(self.varpar_assignment)
	# 					) # E[log q(Z)]
					)
# 			assert output<=0, output
			return output
		else:
			return 0.0
				
	def get_E_log_varpar(self):
		if self.varpar_assignment.size:
			sum_gamma = np.sum(self.varpar_stick, axis=-1)
			digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
			sum_E_log_varpar_label = np.sum(self.varpar_label*np.ma.log(self.varpar_label))
			sum_E_log_varpar_assignment = np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment))
	# 		if sum_E_log_varpar_label is np.ma.masked:
	# 			sum_E_log_varpar_label=0
	# 		if sum_E_log_varpar_assignment is np.ma.masked:
	# 			sum_E_log_varpar_assignment=0
			output= (
					np.sum(
						digamma_diffs[:,0]*(self.varpar_stick[:,0]-1)
						+
						digamma_diffs[:,1]*(self.varpar_stick[:,1]-1)
						-
						np.sum(
							sps.gammaln(self.varpar_stick),
							axis=1
							)
						+
						sps.gammaln(sum_gamma)
						) # E[log q(V)]
					+
					sum_E_log_varpar_label
					+
					sum_E_log_varpar_assignment
					)
# 			assert output<=0, output
			return output
		else:
			return 0.0
		
class DP_bottom(object):
	def __init__(self, T, concent, mother):
		self.mother = mother
# 		num_symbols = num_symbols
		self.varpar_stick = np.random.gamma(10, 0.1, size=(T-1,2)) # gamma in Blei and Jordan.
		self.varpar_label = np.random.dirichlet(np.ones(num_symbols), T) # Emission probability approximated by a multinomial distribution.
		self.concent = concent
		self.T = T
		self.new_id = 0
		self.customers=[]
		self.id = self.mother.add_customer(self)
		
		
	def add_customer(self, target_value):
		issued_id = self.new_id
		self.new_id+=1
		self.customers.append(target_value)
		return issued_id
		
	def set_varpar_assignment(self):
		self.varpar_assignment = np.random.dirichlet(np.ones(self.T), len(self.customers)) # phi in Blei and Jordan.
		self.phi_x_tau_x_error = np.matmul(
								self.varpar_assignment,
								np.matmul(
									self.varpar_label,
									error_mat
									)
								)
		if not self.varpar_assignment.size:
			self.varpar_label=np.zeros(self.varpar_label.shape)
		
	def _update_varpar_stick(self):
# 		print 'varpar_assignment',self.varpar_assignment
# 		print np.sum(self.varpar_assignment, axis=0)[:-1s]
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=0)[:-1]#+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[:,::-1],
											axis=1
												)[:,::-1],
										axis=0)[1:]+(self.concent-1)
		
	def _update_varpar_assignment(self):
		digamma_gamma1plus2 = sps.digamma(np.sum(self.varpar_stick, axis=1))
		self.varpar_assignment = np.exp(
									np.append(sps.digamma(self.varpar_stick[:,0])-digamma_gamma1plus2, 0)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											sps.digamma(self.varpar_stick[:,1])
											-digamma_gamma1plus2
											)
										)[np.newaxis,:]
									+np.matmul(
										error_mat[self.customers], # Nx|∑|
										np.transpose(self.varpar_label) # |∑|xT
										)
								)
		self.varpar_assignment /= np.sum(self.varpar_assignment, axis=-1)[:,np.newaxis]
		
									
											
	def _update_varpar_label(self):
# 		print 'update label'
# 		print np.exp(
# 								np.matmul(
# 									self.mother.varpar_assignment[self.id], # Txmother_T
# 									np.matmul(
# 										self.mother.varpar_label, # mother_T x |∑|
# 										error_mat # |∑|x|∑|
# 										)
# 									)
# 								+np.matmul(
# 									np.transpose(self.varpar_assignment), # TxN
# 									error_mat[self.customers] # Nx|∑|
# 									)
# 								)
# 		print self.varpar_assignment
		self.varpar_label = np.exp(
								np.matmul(
									self.mother.varpar_assignment[self.id], # Txmother_T
									np.matmul(
										self.mother.varpar_label, # mother_T x |∑|
										error_mat # |∑|x|∑|
										)
									)
								+np.matmul(
									np.transpose(self.varpar_assignment), # TxN
									error_mat[self.customers] # Nx|∑|
									)
								)
		self.varpar_label /= np.sum(self.varpar_label, axis=-1)[:, np.newaxis]
								
	def update_varpars(self):
		if self.varpar_assignment.size:
			self._update_varpar_stick()
			assert not np.sum(np.isnan(self.varpar_stick)), ('stick\n',self.varpar_stick)
			self._update_varpar_label()
			assert not np.sum(np.isnan(self.varpar_label)), ('label\n',self.varpar_label)
			self._update_varpar_assignment()
			assert not np.sum(np.isnan(self.varpar_assignment)), ('assignment\n',self.varpar_assignment)
			self.phi_x_tau_x_error = np.matmul(
									self.varpar_assignment,
									np.matmul(
										self.varpar_label,
										error_mat
										)
									)
			assert not np.sum(np.isnan(self.phi_x_tau_x_error)), ('phi_x_tau_x_error\n',self.phi_x_tau_x_error)
		
	def get_sum_E_log_p_parameters(self):
		if self.varpar_assignment.size:
# 			print 'bottom'
			sum_gamma = np.sum(self.varpar_stick, axis=-1)
# 	# 		print 'sum_gamma',sum_gamma
			digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
# 			print 'varpar_stick',self.varpar_stick
# 			print 'varpar_assignment', self.varpar_assignment
# 			print 'digamma_diffs',digamma_diffs
			cumsum_phis = np.cumsum(
								self.varpar_assignment[...,::-1],
								axis=-1
								)[...,::-1]
# 			print 'cumsum_phis',cumsum_phis
# 			print 'E[log p(V | alpha)]',(
# 	# 					digamma_diffs[:,0]
# 	# 					+
# 						digamma_diffs[:,1]*(self.concent-1)
# 						)
# 			print 'E[log p(eta_m | Z_m-1, eta_m-1)]',np.sum(
# 						np.matmul(
# 							self.mother.varpar_assignment[self.id],
# 							self.mother.tau_x_error
# 							)
# 						*
# 						self.varpar_label
# 						)
# 			print 'E[log p(Z | V)]',(
# 						digamma_diffs[:,1]*np.sum(cumsum_phis[:,:-1], axis=0)
# 						+
# 						digamma_diffs[:,0]*np.sum(self.varpar_assignment[:,:-1], axis=0)
# 						)
			output= (
					np.sum(
	# 					digamma_diffs[:,0]
	# 					+
						digamma_diffs[:,1]*(self.concent-1)
						) # E[log p(V | alpha)]
					+np.sum(
						np.matmul(
							np.matmul(
								self.mother.varpar_assignment[self.id],
								self.mother.tau_x_error
								),
							np.transpose(self.varpar_label)
							)
						) # E[log p(eta_m | Z_m-1, eta_m-1)]
					+np.sum(
						digamma_diffs[:,1]*np.sum(cumsum_phis[:,1:], axis=0)
						+
						digamma_diffs[:,0]*np.sum(self.varpar_assignment[:,:-1], axis=0)
						) # E[log p(Z | V)]
	# 				+np.sum(
	# 					digamma_diffs[:,0]*self.varpar_stick[:,0]
	# 					+
	# 					digamma_diffs[:,1]*self.varpar_stick[:,1]
	# 					-
	# 					np.sum(
	# 						sps.gammaln(self.varpar_stick),
	# 						axis=0
	# 						)
	# 					+
	# 					sps.gammaln(sum_gamma)
	# 					) # E[log q(V)]
	# 				+np.sum(
	# 					self.varpar_assignment*np.ma.log(self.varpar_assignment)
	# 					) # E[log q(Z)]
					)
# 			assert output<=0, output
			return output
		else:
			return 0.0
	
		
				
	def get_E_log_varpar(self):
		if self.varpar_assignment.size:
			sum_gamma = np.sum(self.varpar_stick, axis=-1)
			digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
			sum_E_log_varpar_stick = np.sum(
										digamma_diffs[:,0]*(self.varpar_stick[:,0]-1)
										+
										digamma_diffs[:,1]*(self.varpar_stick[:,1]-1)
										-
										np.sum(
											sps.gammaln(self.varpar_stick),
											axis=1
											)
										+
										sps.gammaln(sum_gamma)
										) # E[log q(V)]
# 			assert sum_E_log_varpar_stick<=0, (sum_E_log_varpar_stick,self.varpar_stick) #pdfだから、logが0以下とは限らない。
			sum_E_log_varpar_label = np.sum(self.varpar_label*np.ma.log(self.varpar_label))
			assert sum_E_log_varpar_label<=0, sum_E_log_varpar_label
			sum_E_log_varpar_assignment = np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment))
			assert sum_E_log_varpar_assignment<=0, sum_E_log_varpar_assignment
# 			if sum_E_log_varpar_label is np.ma.masked:
# 				sum_E_log_varpar_label=0
# 			if sum_E_log_varpar_assignment is np.ma.masked:
# 				sum_E_log_varpar_assignment=0
			output= (
					sum_E_log_varpar_stick
					+
					sum_E_log_varpar_label
					+
					sum_E_log_varpar_assignment
					)
# 			assert output<=0, output
			return output
		else:
			return 0	
	
class DP_top(object):
	def __init__(self, T, concent):
		self.varpar_stick = np.random.gamma(10, 0.1, size=(T-1,2)) # gamma in Blei and Jordan.
# 								for i in xrange(n)]
		self.varpar_label = np.random.dirichlet(np.ones(num_symbols), T)
# 								np.random.dirichlet(np.ones(num_symbols), [T]+[num_symbols]*(i))
# 								for i in xrange(n)
# 								]# Emission probability approximated by a multinomial distribution.

		self.concent = concent
		self.T = T
		self.new_id=0
		self.children=[]
		
		
	def add_customer(self, child):
		self.children.append(child)
		issued_id = self.new_id
		self.new_id+=1
		return issued_id
		
	def set_varpar_assignment(self):
# 		self.children = children # Represented by a list of children restaurants.
# 		self.children_label = [child.varpar_label for child in self.children]
		self.num_children = len(self.children)
		self.num_tables_child = self.children[0].varpar_label.shape[0] # T for children.
		self.varpar_assignment = np.random.dirichlet(
											np.ones(self.T),
											(self.num_children, self.num_tables_child)
											) # phi in Blei and Jordan.
		self.tau_x_error = np.matmul(
								self.varpar_label,
								error_mat
								)
		
	def _update_varpar_stick(self):
		self.varpar_stick[...,0] = np.sum(self.varpar_assignment, axis=(0, 1))[:-1]#+1
		self.varpar_stick[...,1] = np.sum(
										np.cumsum(
											self.varpar_assignment[...,::-1],
											axis=-1
												)[...,::-1],
										axis=(0,1)
										)[1:]+(self.concent-1)


	def _update_varpar_assignment(self, children_label_np):
		digamma_gamma1plus2 = sps.digamma(np.sum(self.varpar_stick, axis=1))
		self.varpar_assignment = np.exp(
									np.append(sps.digamma(self.varpar_stick[:,0])-digamma_gamma1plus2, 0)[np.newaxis,:]
									+np.append(
										0,
										np.cumsum(
											sps.digamma(self.varpar_stick[:,1])
											-digamma_gamma1plus2
											)
										)[np.newaxis,:]
									+np.matmul(
										children_label_np,
										np.transpose(self.tau_x_error)
										)
								)
		denom=np.sum(self.varpar_assignment, axis=-1)[:,np.newaxis]
		self.varpar_assignment /= denom+(denom==0).astype(np.float64)
		self.varpar_assignment = self.varpar_assignment.reshape(self.num_children, self.num_tables_child, self.T)
									
											
	def _update_varpar_label(self, children_label_np):
		self.varpar_label = np.exp(
								np.matmul(
									np.matmul(
										np.transpose(self.varpar_assignment.reshape(
																	self.num_children*self.num_tables_child,
																	self.T
																	)
													), # Tx(num_children*num_tables_child)
										children_label_np # (num_children*num_tables_child)x|∑|
										),
									error_mat # |∑|x|∑|
									)
								)
		denom=np.sum(self.varpar_label, axis=-1)[:, np.newaxis]
		self.varpar_label /= denom+(denom==0).astype(np.float64)
		
									
	def update_varpars(self):
		children_label_np = np.array([child.varpar_label for child in self.children]).reshape(
																self.num_children*self.num_tables_child,
																num_symbols
																)# (num_children*num_tables_child)x|∑|
		self._update_varpar_stick()
		assert not np.sum(np.isnan(self.varpar_stick)), ('stick\n',self.varpar_stick)
		self._update_varpar_label(children_label_np)
		assert not np.sum(np.isnan(self.varpar_label)), ('label\n',self.varpar_label)
		self.tau_x_error = np.matmul(
								self.varpar_label,
								error_mat
								)
		assert not np.sum(np.isnan(self.tau_x_error)), ('tau_x_error\n',self.tau_x_error)
		self._update_varpar_assignment(children_label_np)
		assert not np.sum(np.isnan(self.varpar_assignment)), ('assignment\n',self.varpar_assignment)
		
		
	def get_sum_E_log_p_parameters(self):
# 		print 'Top'
		sum_gamma = np.sum(self.varpar_stick, axis=-1)
# 		print 'sum_gamma',sum_gamma
		digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
# 		print 'digamma_diffs',digamma_diffs
		cumsum_phis = np.cumsum(
							self.varpar_assignment[...,::-1],
							axis=-1
							)[...,::-1]
# 		print 'cumsum_phis',cumsum_phis
# 		print 'E[log p(Z | V)]', np.sum(
# # 					digamma_diffs[:,0]
# # 					+
# 					digamma_diffs[:,1]*(self.concent-1)
# 					)
# 		print 'E[log p(Z | V)]', np.sum(
# 					digamma_diffs[:,1]*np.sum(cumsum_phis[...,:-1], axis=(0,1))
# 					+
# 					digamma_diffs[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
# 					)
		output= (
				np.sum(
# 					digamma_diffs[:,0]
# 					+
					digamma_diffs[:,1]*(self.concent-1)
					) # E[log p(V | alpha)]
# 				-np.log(num_symbols)*self.varpar_label.shape[0] # E[log p(eta_1)]=-log(|∑|)<-constant.
				+np.sum(
					digamma_diffs[:,1]*np.sum(cumsum_phis[...,1:], axis=(0,1))
					+
					digamma_diffs[:,0]*np.sum(self.varpar_assignment[...,:-1], axis=(0,1))
					) # E[log p(Z | V)]
# 				+np.sum(
# 					digamma_diffs[:,0]*self.varpar_stick[:,0]
# 					+
# 					digamma_diffs[:,1]*self.varpar_stick[:,1]
# 					-
# 					np.sum(
# 						sps.gammaln(self.varpar_stick),
# 						axis=0
# 						)
# 					+
# 					sps.gammaln(sum_gamma)
# 					) # E[log q(V)]
# 				+np.sum(
# 					self.varpar_assignment*np.ma.log(self.varpar_assignment)
# 					) # E[log q(Z)]
				)
# 		assert output<=0, output
		return output

	def get_E_log_varpar(self):
		sum_gamma = np.sum(self.varpar_stick, axis=-1)
		digamma_diffs = sps.digamma(self.varpar_stick)-sps.digamma(sum_gamma)[:, np.newaxis]
		sum_E_log_varpar_label = np.sum(self.varpar_label*np.ma.log(self.varpar_label))
		sum_E_log_varpar_assignment = np.sum(self.varpar_assignment*np.ma.log(self.varpar_assignment))
# 		if sum_E_log_varpar_label is np.ma.masked:
# 			sum_E_log_varpar_label=0
# 		if sum_E_log_varpar_assignment is np.ma.masked:
# 			sum_E_log_varpar_assignment=0
		output= (
				np.sum(
					digamma_diffs[:,0]*(self.varpar_stick[:,0]-1)
					+
					digamma_diffs[:,1]*(self.varpar_stick[:,1]-1)
					-
					np.sum(
						sps.gammaln(self.varpar_stick),
						axis=1
						)
					+
					sps.gammaln(sum_gamma)
					) # E[log q(V)]
				+
				sum_E_log_varpar_label
				+
				sum_E_log_varpar_assignment
				)
# 		assert output<=0, output
		return output
	
class Word(object):
	def __init__(self, string, n, DP):
		self.ngrams = [Ngram(window)
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
	def __init__(self, window):
		self.restaurants = []
		self.ids = []
		self.context = window[:-1]
		self.target = window[-1]
		
	def enter_a_restaurant(self, base_distr, n):
# 		print self.context,base_distr.tree[n-1]
		restaurant = base_distr.tree[n-1][self.context]
		self.ids.append(restaurant.add_customer(self.target))
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

def code_data(str_data):
	inventory = sorted(set(''.join(str_data)))
	num_symbols = len(inventory)
	coder = {symbol:code for code,symbol in enumerate(inventory, start=2)}
	decoder = {code:symbol for code,symbol in enumerate(inventory)}
	coded_data = [map(lambda s: coder[s],phrase) for phrase in str_data]
# 	print coded_data
	return (coded_data,decoder)

if __name__=='__main__':
# 	warnings.simplefilter('error', UserWarning)
	datapath = sys.argv[1]
	df = pd.read_csv(datapath, sep='\t')
	str_data = list(df.string)
# 	with open(datapath,'r') as f: # Read the phrase file.
# 		str_data = [phrase.replace('\r','\n').strip('\n').split('\t')[0]
# 								for phrase in f.readlines()
# 								]
	customers,decoder = code_data(str_data)
	now = datetime.datetime.now().strftime('%y%m%d%H%M%S')
	result_path = ('./results/'+datapath.split('/')[-1]+now+'/')
	os.makedirs(result_path)
	sl_T = int(sys.argv[2])
	sl_concent = np.float64(sys.argv[3])
	n = int(sys.argv[4])
	T_base = int(sys.argv[5])
	base_concent = np.float64(sys.argv[6])
	noise = np.float64(sys.argv[7])
	max_iters = int(sys.argv[8])
	min_increase = np.float64(sys.argv[9])
	start = datetime.datetime.now()
	vi = VariationalInferenceDP(
			sl_T,
			customers,
			sl_concent,
			n,
			T_base,
			base_concent,
			noise,
			result_path
			)
	vi.run(max_iters, min_increase)
	vi.save_results()
	print 'Time spent',str(datetime.datetime.now()-start)
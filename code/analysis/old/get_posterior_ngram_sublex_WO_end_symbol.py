# coding: utf-8

import numpy as np
import pandas as pd
import scipy.misc as spm
import itertools, os, sys



def get_posterior_ngram(hdf5_path, n, result_path, log_assignment_over_others):
	with pd.HDFStore(hdf5_path, mode='r') as hdf5_store:
		df_unigram_atom = hdf5_store.select('sublex/_1gram/context_/atom')
		for (cluster_id,sublex_id), sub_df_atom in df_unigram_atom.groupby(['cluster_id','sublex_id']):
			normalizer = sub_df_atom.dirichlet_par.sum()
			df_unigram_atom.loc[
				(df_unigram_atom.cluster_id==cluster_id)
				&
				(df_unigram_atom.sublex_id==sublex_id)
				,
				'prob'
				] = sub_df_atom.dirichlet_par / normalizer



		df_stick = hdf5_store.select('sublex/_1gram/context_/stick')
		df_stick['sum'] = df_stick.ix[:,['beta_par1','beta_par2']].sum(axis=1)
		last_cluster = df_stick.cluster_id.max()+1
		num_sublex = df_stick.sublex_id.drop_duplicates().size
		df_stick = df_stick.append(
							pd.DataFrame(
								[
									[sublex_id,last_cluster,1,1,1]
									for sublex_id in range(num_sublex)
								]
								,
								columns = ['sublex_id','cluster_id','beta_par1','beta_par2','sum']
							)
							,
							ignore_index=True
							)

		df_stick['stick_prob'] = df_stick['beta_par1'] / df_stick['sum']
		df_stick['pass_prob'] = df_stick['beta_par2'] / df_stick['sum']
		# print df_stick
		[
			update_stick_prob(df_stick, sublex_id, cluster_id)
			for sublex_id, df_stick_sublex in df_stick.groupby('sublex_id')
			for cluster_id in range(last_cluster)
		]
		# print df_stick#.groupby(['sublex_id']).stick_prob.sum()
		
		df_exp = pd.merge(df_unigram_atom, df_stick, on=['sublex_id', 'cluster_id'], how='outer')
		df_exp['prob'] = df_exp['prob']*df_exp['stick_prob']
		exp_group = pd.DataFrame(df_exp.groupby(['value','sublex_id']).prob.sum()).reset_index()
		# print exp_group.groupby(['sublex_id']).prob.sum()


		df_assignment = hdf5_store.select('sublex/_1gram/context_/assignment')
		df_exp_top_down = pd.merge(df_unigram_atom, df_assignment, on=['sublex_id', 'cluster_id'], how='outer')
		df_exp_top_down['prob'] = df_exp_top_down['p'] * df_exp_top_down['prob']
		top_down_group = pd.DataFrame(df_exp_top_down.groupby(
									['sublex_id','children_DP_context','children_cluster_id','value']
									).prob.sum()
									).reset_index()
		top_down_group = top_down_group.rename(columns={'children_cluster_id':'cluster_id'})
		inventory = sorted(df_unigram_atom.value.drop_duplicates().map(str))
		start_code = str(len(inventory))

		output_filename = os.path.join(result_path, 'posterior_%igram.csv' % n)
		if os.path.exists(output_filename):
			os.remove(output_filename)

		expand(
			'',
			exp_group,
			top_down_group,
			2,
			n,
			inventory,
			start_code,
			hdf5_store,
			output_filename,
			log_assignment_over_others
			)





def expand(
		mother_context,
		mother_exp,
		mother_top_down,
		current_depth,
		full_depth,
		inventory,
		start_code,
		hdf5_store,
		output_filename,
		log_assignment_over_others
		):
	if start_code in mother_context.split('_'):
		prefix_list = [start_code]
	else:
		prefix_list = [start_code]+inventory
	for context_prefix in prefix_list:
		if mother_context:
			context = context_prefix+'_'+mother_context
		else:
			context = context_prefix
		key = ('/sublex/_%igram/context_%s' % (current_depth,context))
		key_stick = os.path.join(key,'stick')
		if key_stick in hdf5_store.keys():
			sub_mother_top_down = mother_top_down[
									mother_top_down.children_DP_context==context
									].drop('children_DP_context', axis=1)

			df_stick = hdf5_store.select(key_stick)
			df_stick['sum'] = df_stick.ix[:,['beta_par1','beta_par2']].sum(axis=1)
			last_cluster = df_stick.cluster_id.max()+1
			num_sublex = df_stick.sublex_id.drop_duplicates().size
			df_stick = df_stick.append(
								pd.DataFrame(
									[
										[sublex_id,last_cluster,1,1,1]
										for sublex_id in range(num_sublex)
									]
									,
									columns = ['sublex_id','cluster_id','beta_par1','beta_par2','sum']
								),
								ignore_index=True
								)

			df_stick['stick_prob'] = df_stick['beta_par1'] / df_stick['sum']
			df_stick['pass_prob'] = df_stick['beta_par2'] / df_stick['sum']
			[
				update_stick_prob(df_stick, sublex_id, cluster_id)
				for sublex_id, df_stick_sublex in df_stick.groupby('sublex_id')
				for cluster_id in range(last_cluster)
			]

			
			df_exp = pd.merge(sub_mother_top_down, df_stick, on=['sublex_id', 'cluster_id'], how='outer')
			df_exp['prob'] = df_exp['prob']*df_exp['stick_prob']
			exp_group = pd.DataFrame(df_exp.groupby(['sublex_id','value']).prob.sum()).reset_index()
			# print exp_group#.groupby(['sublex_id']).prob.sum()
			
			
			if current_depth == full_depth:
				exp_group['context'] = context
				exp_group['context_in_data'] = True
				save_posterior_ngram(exp_group, log_assignment_over_others, output_filename)
			else:
				df_assignment = hdf5_store.select(os.path.join(key,'assignment'))

				df_exp_top_down = pd.merge(sub_mother_top_down, df_assignment, on=['sublex_id', 'cluster_id'], how='outer')
				df_exp_top_down['prob'] = df_exp_top_down['p'] * df_exp_top_down['prob']
				top_down_group = pd.DataFrame(df_exp_top_down.groupby(
									['sublex_id','children_DP_context','children_cluster_id','value']
									).prob.sum()
									).reset_index()
				top_down_group = top_down_group.rename(columns={'children_cluster_id':'cluster_id'})
				expand(
					context,
					exp_group,
					top_down_group,
					current_depth+1,
					full_depth,
					inventory,
					start_code,
					hdf5_store,
					output_filename,
					log_assignment_over_others
					)
		else:
			mother_copy = mother_exp.copy()
			mother_copy['context'] = context
			if current_depth == full_depth:
				mother_copy['context_in_data'] = False
				save_posterior_ngram(mother_copy, log_assignment_over_others, output_filename)
			else:
				expand(
					context,
					mother_copy,
					None,
					current_depth+1,
					full_depth,
					inventory,
					start_code,
					hdf5_store,
					output_filename,
					log_assignment_over_others
					)

def update_stick_prob(df_stick, sublex_id, cluster_id):
	df_stick.loc[
				(df_stick.sublex_id == sublex_id)
				&
				(df_stick.cluster_id > cluster_id),
				'stick_prob'
				] *= df_stick.loc[
						(df_stick.sublex_id == sublex_id)
						&
						(df_stick.cluster_id == cluster_id)
						,
						'pass_prob'
						].iat[0]

def save_posterior_ngram(ngram_per_context, log_assignment_over_others, filename):
	# print ngram_per_context.groupby(['sublex_id','context']).prob.sum()
	ngram_per_context = ngram_per_context.sort_values('sublex_id')
	[
		get_representativeness(
			ngram_per_context,
			log_assignment_over_others,
			value,
			df_per_value
			)
		for value, df_per_value
		in ngram_per_context.groupby('value')
	]
	ngram_per_context.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))


def get_representativeness(
			df_ngram,
			log_assignment_over_others,
			value,
			df_per_value,
			):
	log_ngram_prob_x_assignment = (
									np.log(df_per_value.prob)[np.newaxis,:]
									+
									log_assignment_over_others
									)
	# np.fill_diagonal(log_ngram_prob_x_assignment, np.float64(1))
	log_denominator = spm.logsumexp(log_ngram_prob_x_assignment, axis=1)
	df_ngram.loc[
		df_ngram.value == value
		,
		'representativeness'
		] = np.log(df_per_value.prob) - log_denominator



if __name__=='__main__':
	hdf5_path = sys.argv[1]
	n = int(sys.argv[2])
	result_path = os.path.split(hdf5_path)[0]

	df_stick = pd.read_hdf(hdf5_path, key='/sublex/stick')
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


	ngram = get_posterior_ngram(hdf5_path, n, result_path, log_assignment_over_others)
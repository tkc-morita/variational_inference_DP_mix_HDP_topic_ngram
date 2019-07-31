# coding: utf-8

import numpy as np
import pandas as pd
import argparse
import scipy.stats as sps
import sklearn.metrics as skm
# import my_autopct

def bootstrap(df, num_iters, seed=111):
	random_state = np.random.RandomState(seed)
	f_samples = {
		"vs_model":{
			"gram_as_pos":{
				"precision":[],
				"recall":[],
				"f":[]
			},
			"ungram_as_pos":{
				"precision":[],
				"recall":[],
				"f":[]
			},
		},
		"vs_etym":{
			"gram_as_pos":{
				"precision":[],
				"recall":[],
				"f":[]
			},
			"ungram_as_pos":{
				"precision":[],
				"recall":[],
				"f":[]
			},
		},
	}
	v_samples = {
		"vs_model":[],
		"vs_etym":[]
	}
	for iter_id in range(num_iters):
		df_resampled = df.sample(frac=1.0, replace=True, random_state=random_state)
		f_results, v_results = get_f_and_v_scores(df_resampled)
		f_samples["vs_model"]["gram_as_pos"]["precision"].append(f_results["vs_model"]["gram_as_pos"]["precision"])
		f_samples["vs_model"]["gram_as_pos"]["recall"].append(f_results["vs_model"]["gram_as_pos"]["recall"])
		f_samples["vs_model"]["gram_as_pos"]["f"].append(f_results["vs_model"]["gram_as_pos"]["f"])

		f_samples["vs_model"]["ungram_as_pos"]["precision"].append(f_results["vs_model"]["ungram_as_pos"]["precision"])
		f_samples["vs_model"]["ungram_as_pos"]["recall"].append(f_results["vs_model"]["ungram_as_pos"]["recall"])
		f_samples["vs_model"]["ungram_as_pos"]["f"].append(f_results["vs_model"]["ungram_as_pos"]["f"])

		f_samples["vs_etym"]["gram_as_pos"]["precision"].append(f_results["vs_etym"]["gram_as_pos"]["precision"])
		f_samples["vs_etym"]["gram_as_pos"]["recall"].append(f_results["vs_etym"]["gram_as_pos"]["recall"])
		f_samples["vs_etym"]["gram_as_pos"]["f"].append(f_results["vs_etym"]["gram_as_pos"]["f"])

		f_samples["vs_etym"]["ungram_as_pos"]["precision"].append(f_results["vs_etym"]["ungram_as_pos"]["precision"])
		f_samples["vs_etym"]["ungram_as_pos"]["recall"].append(f_results["vs_etym"]["ungram_as_pos"]["recall"])
		f_samples["vs_etym"]["ungram_as_pos"]["f"].append(f_results["vs_etym"]["ungram_as_pos"]["f"])

		v_samples["vs_model"].append(v_results["vs_model"])
		v_samples["vs_etym"].append(v_results["vs_etym"])
	return f_samples, v_samples


def get_f_and_v_scores(df):
	# vs. model predictions
	f_results = {
		"vs_model":{
			"gram_as_pos":{},
			"ungram_as_pos":{},
		},
		"vs_etym":{
			"gram_as_pos":{},
			"ungram_as_pos":{},
		},
	}
	v_results = {}
	precision, recall, f = get_f_score(
		df.is_grammatical.astype(int),
		(~df.is_sublex_2).astype(int)
		)
	f_results["vs_model"]["gram_as_pos"]["precision"] = precision
	f_results["vs_model"]["gram_as_pos"]["recall"] = recall
	f_results["vs_model"]["gram_as_pos"]["f"] = f


	precision, recall, f = get_f_score(
		(~df.is_grammatical).astype(int),
		df.is_sublex_2.astype(int)
	)
	f_results["vs_model"]["ungram_as_pos"]["precision"] = precision
	f_results["vs_model"]["ungram_as_pos"]["recall"] = recall
	f_results["vs_model"]["ungram_as_pos"]["f"] = f

	v_results["vs_model"] = skm.v_measure_score(
		df.is_grammatical.astype(int),
		df.is_sublex_2.astype(int)
	)
	

	precision, recall, f = get_f_score(
		df.is_grammatical.astype(int),
		(~df.is_Latin).astype(int)
		)
	f_results["vs_etym"]["gram_as_pos"]["precision"] = precision
	f_results["vs_etym"]["gram_as_pos"]["recall"] = recall
	f_results["vs_etym"]["gram_as_pos"]["f"] = f


	precision, recall, f = get_f_score(
		(~df.is_grammatical).astype(int),
		df.is_Latin.astype(int)
	)
	f_results["vs_etym"]["ungram_as_pos"]["precision"] = precision
	f_results["vs_etym"]["ungram_as_pos"]["recall"] = recall
	f_results["vs_etym"]["ungram_as_pos"]["f"] = f

	v_results["vs_etym"] = skm.v_measure_score(
		df.is_grammatical.astype(int),
		df.is_Latin.astype(int)
	)

	f_results["vs_GandP"] = {
		"gram_as_pos":{},
		"ungram_as_pos":{}
	}
	precision, recall, f = get_f_score(
		df.is_grammatical.astype(int),
		(~df.subject2GrimshawPrinceConstraint).astype(int)
		)
	f_results["vs_GandP"]["gram_as_pos"]["precision"] = precision
	f_results["vs_GandP"]["gram_as_pos"]["recall"] = recall
	f_results["vs_GandP"]["gram_as_pos"]["f"] = f


	precision, recall, f = get_f_score(
		(~df.is_grammatical).astype(int),
		df.subject2GrimshawPrinceConstraint.astype(int)
	)
	f_results["vs_GandP"]["ungram_as_pos"]["precision"] = precision
	f_results["vs_GandP"]["ungram_as_pos"]["recall"] = recall
	f_results["vs_GandP"]["ungram_as_pos"]["f"] = f

	v_results["vs_GandP"] = skm.v_measure_score(
		df.is_grammatical.astype(int),
		df.subject2GrimshawPrinceConstraint.astype(int)
	)

	return f_results, v_results

def get_f_score(grand_truth, predictions):
	precision = skm.precision_score(grand_truth, predictions)
	recall = skm.recall_score(grand_truth, predictions)
	f = sps.hmean([precision, recall])
	return precision, recall, f

def get_random_baselines(grand_truth, num_iters, p=(0.5,0.5), seed=111, method='uniform'):
	random_state = np.random.RandomState(seed)
	fs = []
	fs_flipped = []
	vs = []
	for iter_id in range(num_iters):
		if method=='uniform':
			samples = random_state.randint(2, size=grand_truth.size)
		if method=='proportional':
			samples = random_state.choice(np.arange(2), size=grand_truth.size, p=p)
		if method=='shuffle':
			samples = random_state.permutation(grand_truth)
		fs.append(skm.f1_score(grand_truth, samples))
		fs_flipped.append(skm.f1_score(1-grand_truth, 1-samples))
		vs.append(skm.v_measure_score(grand_truth, samples))
	return fs, fs_flipped, vs


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the tsv file containing full data classified.')
	parser.add_argument('--num_iters', type=int, default=100000, help='# of iterations for bootstrap sampling.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	args = parser.parse_args()

	df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')

	print('Excluded:')
	print(df[df.most_probable_sublexicon.isnull()])

	print('Mixed:')
	print(df[df.MyEtym=='MIXED'])

	df = df[~df.most_probable_sublexicon.isnull()]
	df = df[df.MyEtym.isin(['Latin','non_Latin'])]
	print(df.MyEtym.value_counts())
	
	# linguistic_work = 'Levin1993_Latin'
	# linguistic_work = 'YangMontrul2017_Latin'
	# linguistic_work = 'YangMontrul2017_none'
	# my_etym = 'non_Latin'
	# df = df[df.linguistic_work == linguistic_work]
	# df = df[df.MyEtym == my_etym]

	# remove_circumfix = lambda s: int(s.split('_')[1])
	df['is_grammatical'] = (df.double_object == 'grammatical')
	# df['predicted_sublex'] = df_pred.most_probable_sublexicon.map(remove_circumfix)
	df['is_sublex_2'] = (df.most_probable_sublexicon=='sublex_2')
	df['is_Latin'] = (df.MyEtym=='Latin')

	f_results, v_results = get_f_and_v_scores(df)
	f_uniform_gram_as_pos, f_uniform_ungram_as_pos, v_uniform = get_random_baselines(
		df.is_grammatical.astype(int),
		args.num_iters,
		seed=args.seed,
		method='uniform'
	)
	f_proportional_gram_as_pos, f_proportional_ungram_as_pos, v_proportional = get_random_baselines(
		df.is_grammatical.astype(int),
		args.num_iters,
		seed=args.seed,
		method='proportional'
	)
	f_shuffle_gram_as_pos, f_shuffle_ungram_as_pos, v_shuffle = get_random_baselines(
		df.is_grammatical.astype(int),
		args.num_iters,
		seed=args.seed,
		method='shuffle'
	)
	random_baselines = {
		"uniform":{
			'gram_as_pos':f_uniform_gram_as_pos,
			'ungram_as_pos':f_uniform_ungram_as_pos,
			'v':v_uniform,
		},
		'proportional':{
			'gram_as_pos':f_proportional_gram_as_pos,
			'ungram_as_pos':f_proportional_ungram_as_pos,
			'v':v_proportional,
		},
		'shuffle':{
			'gram_as_pos':f_shuffle_gram_as_pos,
			'ungram_as_pos':f_shuffle_ungram_as_pos,
			'v':v_shuffle,
		}
	}
	# f_samples, v_samples = bootstrap(df, args.num_iters, seed=args.seed)
	# f_model_minus_etm = {
	# 		"gram_as_pos":{
	# 			"precision":0.0,
	# 			"recall":0.0,
	# 			"f":0.0
	# 		},
	# 		"ungram_as_pos":{
	# 			"precision":0.0,
	# 			"recall":0.0,
	# 			"f":0.0
	# 		}
	# }
	# v_model_minus_etm = 0.0
	for model_or_etym, sub_f_results in f_results.items():
		print(model_or_etym)
		# sub_f_samples = f_samples[model_or_etym]
		v_score = v_results[model_or_etym]
		# v_sampled_values = v_samples[model_or_etym]
		for pos_type, subsub_f_results in sub_f_results.items():
			print(pos_type)
			# subsub_f_samples = sub_f_samples[pos_type]
			for value_type, value in subsub_f_results.items():
				# sampled_values = subsub_f_samples[value_type]
				# print("{value_type}: {value:.4f} (0.025%:{low:.4f}, 0.0925:{high:.4f})".format(
				print("{value_type}: {value:.4f}".format(
					value_type=value_type,
					value=value,
					# low=np.percentile(sampled_values, 2.5),
					# high=np.percentile(sampled_values, 97.5)
				))
				# if model_or_etym=='vs_model':
				# 	f_model_minus_etm[pos_type][value_type] += np.array(sampled_values)
				# else:
				# 	f_model_minus_etm[pos_type][value_type] -= np.array(sampled_values)
			print('F-score p-values')
			for method,samples in random_baselines.items():
				print('{method}: {p:.6f}'.format(method=method, p=(subsub_f_results['f'] < samples[pos_type]).mean()))
		# print("V-measure score: {value:.4f} (0.025%:{low:.4f}, 0.0925:{high:.4f})".format(
		print("V-measure score: {value:.4f}".format(
					value=v_score,
					# low=np.percentile(v_sampled_values, 2.5),
					# high=np.percentile(v_sampled_values, 97.5)
				))
		print("V-measure score p-values")
		for method,samples in random_baselines.items():
			print('{method}: {p:.6f}'.format(method=method, p=(v_score < samples['v']).mean()))
	# 	if model_or_etym=='vs_model':
	# 		v_model_minus_etm += np.array(v_sampled_values)
	# 	else:
	# 		v_model_minus_etm -= np.array(v_sampled_values)
	# for pos_type, sub_f_model_minus_etm in f_model_minus_etm.items():
	# 	print(pos_type)
	# 	for value_type, array in sub_f_model_minus_etm.items():
	# 		print("model > etymology ({value_type}): {value:.6f}".format(
	# 				value_type=value_type,
	# 				value=(array>0).sum() / float(args.num_iters),
	# 			))
	# print("model > etymology (V-measure score): {value:.6f}".format(
	# 				value=(v_model_minus_etm>0).sum()/ float(args.num_iters),
	# 			))


	# grammatical = (df.double_object == 'grammatical')
	# ungrammatical = (df.double_object == 'ungrammatical')

	# num_grammatical = grammatical.astype(float).sum()
	# num_ungrammatical = ungrammatical.astype(float).sum()


	# print('===================Based on predicted sublexica==================')
	# map_sublex_2 = (df.most_probable_sublexicon == 'sublex_2')
	# map_sublex_5 = (df.most_probable_sublexicon == 'sublex_5')

	# grammatical_sublex_5 = (grammatical & map_sublex_5)
	# ungrammatical_sublex_2 = (ungrammatical & map_sublex_2)

	# map_sublex_2_size = map_sublex_2.astype(float).sum()
	# map_sublex_5_size = map_sublex_5.astype(float).sum()

	# num_grammatical_sublex_5 = grammatical_sublex_5.astype(float).sum()
	# num_ungrammatical_sublex_2 = ungrammatical_sublex_2.astype(float).sum()


	# print('Grammatical as True Positive')
	# precision = num_grammatical_sublex_5 / map_sublex_5_size
	# print('precision', precision)
	# recall = num_grammatical_sublex_5 / num_grammatical
	# print('recall', recall)
	# f = sps.hmean([precision, recall])
	# print('f', f)

	# print('Ungrammatical as True Positive')
	# precision = num_ungrammatical_sublex_2 / map_sublex_2_size
	# print('precision', precision)
	# recall = num_ungrammatical_sublex_2 / num_ungrammatical
	# print('recall', recall)
	# f = sps.hmean([precision, recall])
	# print('f', f)

	# print('V-measure')
	# v = 



	# # Based on true sublexica
	# print('===================Based on true sublexica==================')

	# latin = (df.MyEtym == 'Latin')
	# non_latin = (df.MyEtym == 'non_Latin')

	# grammatical_non_latin = (grammatical & non_latin)
	# ungrammatical_latin = (ungrammatical & latin)

	# latin_size = latin.astype(float).sum()
	# non_latin_size = non_latin.astype(float).sum()

	# num_grammatical_non_latin = grammatical_non_latin.astype(float).sum()
	# num_ungrammatical_latin = ungrammatical_latin.astype(float).sum()


	# print('Grammatical as True Positive')
	# precision = num_grammatical_non_latin / non_latin_size
	# print('precision', precision)
	# recall = num_grammatical_non_latin / num_grammatical
	# print('recall', recall)
	# f = sps.hmean([precision, recall])
	# print('f', f)

	# print('Ungrammatical as True Positive')
	# precision = num_ungrammatical_latin / latin_size
	# print('precision', precision)
	# recall = num_ungrammatical_latin / num_ungrammatical
	# print('recall', recall)
	# f = sps.hmean([precision, recall])
	# print('f', f)
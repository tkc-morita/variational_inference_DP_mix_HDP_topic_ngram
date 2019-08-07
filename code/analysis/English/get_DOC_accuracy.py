# coding: utf-8

import numpy as np
import pandas as pd
import argparse
import scipy.stats as sps
import sklearn.metrics as skm
# import my_autopct

def bootstrap(df, num_iters, seed=111):
	random_state = np.random.RandomState(seed)
	accuracy_samples = {
		"vs_model":[],
		"vs_etym":[],
		"vs_GandP":[]
	}
	for iter_id in range(num_iters):
		df_resampled = df.sample(frac=1.0, replace=True, random_state=random_state)
		accuracy = get_f_and_v_scores(df_resampled)
		for predictor, score in accuracy.items():
			accuracy_samples[predictor].append(score)
	return accuracy_samples


def get_accuracy(df):
	accuracy = {}
	accuracy["vs_model"] = (df.is_grammatical==(~df.is_sublex_2)).mean()
	accuracy["vs_etym"] = (df.is_grammatical==(~df.is_Latin)).mean()
	accuracy["vs_GandP"] = (df.is_grammatical==(~df.subject2GrimshawPrinceConstraint)).mean()
	return accuracy


def get_random_baselines(ground_truth, num_iters, p=(0.5,0.5), seed=111, method='uniform'):
	random_state = np.random.RandomState(seed)
	scores = []
	for iter_id in range(num_iters):
		if method=='uniform':
			samples = random_state.randint(2, size=ground_truth.size)
		if method=='proportional':
			samples = random_state.choice(np.arange(2), size=ground_truth.size, p=p)
		if method=='shuffle':
			samples = random_state.permutation(ground_truth)
		scores.append((samples==ground_truth).mean())
	return scores


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

	accuracy = get_accuracy(df)
	uniform_samples = get_random_baselines(
		df.is_grammatical.astype(int),
		args.num_iters,
		seed=args.seed,
		method='uniform'
	)
	proportional_samples = get_random_baselines(
		df.is_grammatical.astype(int),
		args.num_iters,
		seed=args.seed,
		method='proportional'
	)
	shuffle_samples = get_random_baselines(
		df.is_grammatical.astype(int),
		args.num_iters,
		seed=args.seed,
		method='shuffle'
	)
	for predictor, score in accuracy.items():
		print(predictor)
		print('Accuracy: {:0.6f}'.format(score))
		print('p-value (uniform): {:0.6f}'.format((score <= uniform_samples).mean()))
		print('p-value (proportional): {:0.6f}'.format((score <= proportional_samples).mean()))
		print('p-value (shuffle): {:0.6f}'.format((score <= shuffle_samples).mean()))

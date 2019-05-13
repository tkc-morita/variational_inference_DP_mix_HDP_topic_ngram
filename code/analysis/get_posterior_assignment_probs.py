# coding: utf-8

import numpy as np
import pandas as pd
import posterior_predictive_inferences as ppi
import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('hdf_path', type=str, help='Path to the hdf5 file containing parameters of the trained model.')
	args = parser.parse_args()
	
	df_stick = pd.read_hdf(args.hdf_path, key='/sublex/stick')
	log_assignment_probs = ppi.get_log_assignment_probs(df_stick)
	assignment_probs = np.exp(log_assignment_probs)

	print('Assignment probs.')
	for sublex_ix,p in enumerate(assignment_probs):
		print('{sublex_ix}: {p}'.format(sublex_ix=sublex_ix, p=p))

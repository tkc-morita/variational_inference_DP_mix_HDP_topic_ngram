# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path


def plot_ELBO(df, result_dir):
	df.plot(x='iteration',y='ELBO',legend=False)
	plt.title('History of ELBO')
	plt.ylabel('ELBO')
	plt.tight_layout()
	# plt.show()
	plt.savefig(os.path.join(result_dir,'ELBO_history.png'), bbox_inches='tight')

def extract_ELBO(path):
	elbos = []
	with open(path, 'r') as f:
		for line in f.readlines():
			if 'Current var_bound is ' in line:
				elbos.append(np.float64(line.split('Current var_bound is ')[1].split(' (')[0]))
	return elbos


if __name__ == '__main__':
	path = sys.argv[1]

	elbo_constant = np.float64(sys.argv[2])

	df = pd.DataFrame()
	df['ELBO'] = extract_ELBO(path)
	df['ELBO'] += elbo_constant
	df['iteration'] = df.index + 1

	plot_ELBO(df, os.path.split(path)[0])

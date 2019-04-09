# coding: utf-8

import pandas as pd
import sys, os.path, glob


def get_df(path):
	df = pd.DataFrame()
	for filepath in glob.glob(os.path.join(path, 'final_var_bounds*.csv')):
		mini_df = pd.read_csv(filepath)
		mini_df['filename'] = os.path.split(filepath)[1]
		df = df.append(mini_df, ignore_index=True)
	return df

def print_top_k(df, k):
	print df.sort_values('final_var_bound', ascending=False).head(n=k)
	print '# of trials: %i' % df.shape[0]

if __name__ == '__main__':
	path = sys.argv[1]
	k = int(sys.argv[2])
	df = get_df(path)
	print_top_k(df, k)
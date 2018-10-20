# coding: utf-8

import pandas as pd
import sys, os.path

if __name__ == '__main__':
	path = sys.argv[1]

	df = pd.read_csv(path)
	df_dropped = df.dropna().reset_index(drop=True)
	df_dropped.loc[:,'new_id'] = df_dropped.index
	root,ext = os.path.splitext(path)
	df_dropped.to_csv(root+'_unfinished_dropped'+ext, index=False)
# coding: utf-8

import pandas as pd
import os.path, sys

def reshape_hdf(hdf_path):
	result_path,result_filename = os.path.split(hdf_path)
	result_root = os.path.splitext(result_filename)[0]
	with pd.HDFStore(
				hdf_path,
				mode='r'
				) as hdf5_store_old, pd.HDFStore(
				os.path.join(result_path,result_root+'_reshaped.h5')
				) as hdf5_store_new:
		for key in hdf5_store_old.keys():
			if not ('3gram' in key and 'assignment' in key):
				hdf5_store_new[key] = hdf5_store_old.get(key)

	

if __name__ == '__main__':
	hdf_path = sys.argv[1]
	reshape_hdf(hdf_path)
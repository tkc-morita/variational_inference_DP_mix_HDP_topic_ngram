# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

def plot_core_frequency(df):
	plt.figure()
	df.core_frequency.hist(bins=100)
	plt.savefig("core_noun_core_freqs.png")
	
	
if __name__=='__main__':
	data_path='../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns.tsv'
	df=pd.read_csv(data_path, encoding='utf-8', sep='\t')
	print df.shape[0]
	df.core_frequency.value_counts().to_csv("core_freqs.csv")
# 	plot_core_frequency(df)
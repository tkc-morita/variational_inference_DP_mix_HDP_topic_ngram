# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path

def plot_rendaku_rate_against_Nativeness(df):
	df.plot.scatter(x='rendaku_rate_kojien', y='sublex_4')
	plt.show()

def kanji2alph(class_name):
	in_eng = class_name.replace(u'漢', 'SJ').replace(u'和','Native').replace(u'混', 'Mixed').replace(u'外', 'Foreign').replace(u'固', 'Proper').replace(u'記号', 'Symbols').replace(u':','-')
	if 'SJ' in in_eng and 'Native' in in_eng:
		in_eng = 'Mixed'
	else:
		in_eng = in_eng.split('-')[0].rstrip('123')
	return in_eng

def is_rendaku_applicable(feature_word):
	return (startswith_voiceless(feature_word) and not contains_voiced_obstruent(feature_word))

def startswith_voiceless(feature_word):
	initial_segment = feature_word.split(',')[0]
	return initial_segment[5] == 'f'

def contains_voiced_obstruent(feature_word):
	for segment in feature_word.split(','):
		if segment[3] in ['t','s','z'] and segment[5] == 'v':
			return True
	return False

if __name__ == '__main__':
	data_path = sys.argv[2]
	result_path = sys.argv[1]
	feature_path = '../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_5th-ed.tsv'

	df_data = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	df_data[['features','pos']] = pd.read_csv(feature_path, sep='\t', encoding='utf-8')[['features','pos']]
	df_result = pd.read_csv(result_path, encoding='utf-8')

	df = pd.concat([df_data, df_result], axis=1)
	df['actual_sublex']=df.wType.map(lambda x: kanji2alph(x))
	df = df[df.pos.str.startswith(u'名') & df.features.map(is_rendaku_applicable)]

	# df = df[(df.actual_sublex=='SJ')]

	print 'overall'
	print df.loc[:,['rendaku_rate_kojien','sublex_4']].corr(method='spearman')
	plot_rendaku_rate_against_Nativeness(df)
	print 'Native'
	print df.loc[df.actual_sublex=='Native',['rendaku_rate_kojien','sublex_4']].corr(method='spearman')
	plot_rendaku_rate_against_Nativeness(df.loc[df.actual_sublex=='Native'])
	print 'SJ'
	print df.loc[df.actual_sublex=='SJ',['rendaku_rate_kojien','sublex_4']].corr(method='spearman')
	plot_rendaku_rate_against_Nativeness(df.loc[df.actual_sublex=='SJ'])

	

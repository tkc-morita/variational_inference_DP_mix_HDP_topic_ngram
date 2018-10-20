# coding: utf-8

import pandas as pd
import sklearn.metrics as skm
import sys, os


def get_V_measure(df):
	labels_true = df.actual_sublex
	labels_pred = df.predicted_sublex

	return skm.v_measure_score(labels_true, labels_pred)


def get_correct_sublex():
	df_data = pd.read_csv("../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_5th-ed.tsv", sep='\t', encoding='utf-8')
	kanji2ix=dict([(u'和', 0), (u'漢', 1), (u'外', 2), (u'混', 3), (u'固', 4), (u'記号', 5)])
	df_data['actual_sublex']=df_data.wType.map(kanji2ix)
	return df_data




	


if __name__ == '__main__':
	filepath = sys.argv[1]
	df_pred = pd.read_csv(filepath)
	

	df = get_correct_sublex()
	remove_circumfix = lambda s: int(s.split('_')[1])
	df['predicted_sublex'] = df_pred.most_probable_sublexicon.map(remove_circumfix)

	df = df[df.actual_sublex.isin(range(3))]
	
	v_measure = get_V_measure(df)

	result_dir = os.path.split(filepath)[0]
	with open(os.path.join(result_dir,'v-measure_Native-SJ-Foreign.txt'), 'w') as f:
		f.write('V-measure: %s' % str(v_measure))
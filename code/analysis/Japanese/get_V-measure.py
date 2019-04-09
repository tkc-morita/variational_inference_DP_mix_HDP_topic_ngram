# coding: utf-8

import pandas as pd
import sklearn.metrics as skm
import argparse, os


def get_V_measure(df):
	labels_true = df.actual_sublex
	labels_pred = df.predicted_sublex

	return skm.v_measure_score(labels_true, labels_pred)


def get_correct_sublex(data_path):
	df_data = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	kanji2ix=dict([(u'和', 0), (u'漢', 1), (u'外', 2), (u'混', 3), (u'固', 4), (u'記号', 5)])
	df_data['actual_sublex']=df_data.wType.map(kanji2ix)
	return df_data




	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('result_path', type=str, help='Path to the classification results.')
	parser.add_argument('data_path', type=str, help='Path to the data, containing the grand truth classification info.')
	parser.add_argument('-w', '--whole_data', action='store_true', help='Use the whole data rather than Native, SJ, Foreign alone.')
	args = parser.parse_args()


	df_pred = pd.read_csv(args.result_path)
	

	df = get_correct_sublex(args.data_path)
	remove_circumfix = lambda s: int(s.split('_')[1])
	df['predicted_sublex'] = df_pred.most_probable_sublexicon.map(remove_circumfix)

	if args.whole_data:
		filename = 'v-measure_whole-data.txt'
	else:
		filename = 'v-measure_Native-SJ-Foreign.txt'
		df = df[df.actual_sublex.isin(range(3))]
	
	v_measure = get_V_measure(df)

	result_dir = os.path.split(args.result_path)[0]
	with open(os.path.join(result_dir,filename), 'w') as f:
		f.write('V-measure: %s' % str(v_measure))
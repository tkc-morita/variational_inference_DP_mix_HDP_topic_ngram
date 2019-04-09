# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path

def list_representative_words(
		df,
		sublex_id,
		# true_sublex,
		list_length,
		result_dir
		):
	# df = df[df.actual_sublex==true_sublex]
	df = df.sort_values('sublex_%i' % sublex_id, ascending=False)
	df.head(n=list_length).to_csv(os.path.join(result_dir, 'top-%i-representative_words_with-respect-to-sublex-%i.csv' % (list_length, sublex_id)), index=False, encoding='utf-8')
	df.tail(n=list_length).to_csv(os.path.join(result_dir, 'worst-%i-representative_words_with-respect-to-sublex-%i.csv' % (list_length, sublex_id)), index=False, encoding='utf-8')



if __name__ == '__main__':
	result_path = sys.argv[1]
	result_dir, filename = os.path.split(result_path)
	# n = int(filename.split('gram')[0][-1])

	sublex_id = int(sys.argv[4])
	# true_sublex = sys.argv[5]

	list_length = int(sys.argv[3])
	
	df = pd.read_csv(result_path)
	datapath = sys.argv[2]
	df_data = pd.read_csv(datapath, sep='\t', encoding='utf-8')
	df['IPA'] = df_data.IPA
	df['katakana'] = df_data.lForm
	df['orthography'] = df_data.lemma
	df['sub_orthography'] = df_data.subLemma

	df_assignment = pd.read_csv(os.path.join(result_dir, 'SubLexica_assignment.csv'))
	df['MAP_sublex'] = df_assignment.most_probable_sublexicon

	kanji2alph=dict([(u'漢', 'SJ'), (u'和','Native'), (u'混', 'Mixed'), (u'外', 'Foreign'), (u'固', 'Proper'), (u'記号', 'Symbols')])
	df['actual_sublex']=df_data.wType.map(kanji2alph)

	list_representative_words(
		df,
		sublex_id,
		# true_sublex,
		list_length,
		result_dir
		)






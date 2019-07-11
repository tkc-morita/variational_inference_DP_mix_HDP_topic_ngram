# coding: utf-8

import pandas as pd
import pytablewriter, sys

def dummy_code_special_symbols(pd_series):
	return pd_series.replace('r', u'ɾ').str.replace('9',u'ɰ').str.replace('g',u'ɡ')

def utf8_symbols_to_latex_commands(latex_table, sublex_id=''):
	return latex_table.replace(
								u'ɲ', r'{\textltailn}'
							).replace(
								u'ç', r'\c{c}'
							).replace(
								u'ʦ', r'{\texttslig}'
							).replace(
								u'ː', r'{\textlengthmark}'
							).replace(
								u'ɾ', r'{\textfishhookr}'
							).replace(
								u'ɯ', r'{\textturnm}'
							).replace(
								u'ʨ', r'{\texttctclig}'
							).replace(
								u'ʣ', r'{\textdzlig}'
							).replace(
								u'ɕ', r'{\textctc}'
							).replace(
								u'ŋ', r'\textipa{N}'
							).replace(
								u'ɰ', r'\~{\textturnmrleg}'
							).replace(
								u'ɡ', r'{\textscriptg}'
							).replace(
								u'ʥ', r'{\textdctzlig}'
							).replace(
								u'ä', r'a'
							).replace(
								u'ɴ', r'{\textscn}'
							).replace(
								u'ɸ', r'{\textphi}'
							).replace(
								u'START', r'\texttt{START}'
							).replace(
								u'END', r'\texttt{END}'
							).replace(
								u'context', r'$\mathbf{u}$'
							).replace(
								u'value', r'$x_{\mathrm{new}}$'
							).replace(
								u'rep.', r'$R(x_{\mathrm{new}}, \mathbf{u}, '+str(sublex_id)+r')$'
							).replace(
								u'frequency', r'freq.'
							)

def print_utf8(utf8_string):
	print utf8_string.encode('utf-8')

if __name__ == '__main__':
	path = sys.argv[1]
	df = pd.read_csv(path, encoding='utf-8')

	# For ngram rep/
	sublex_id = int(path.split('sublex-')[1][0]) # int(sys.argv[2])
	df = df.drop(labels=['sublex_id','prob'], axis=1)
	df.loc[:,'context'] = dummy_code_special_symbols(df.decoded_context).str.replace('_',',')
	df.loc[:,'value'] = dummy_code_special_symbols(df.decoded_value)
	# df = df.drop(labels = [
	# 						'sublex_id',
	# 						'decoded_value',
	# 						'decoded_context',
	# 						'prob',
	# 						'context_in_data',
	# 						'frequency',
	# 						'expected_frequency'
	# 						], axis=1)
	df.loc[:,'rep.'] = df.representativeness
	df = df[['context', 'value', 'rep.', 'frequency']]

	# For probable words
	# sublex_id = int(path.split('sublex-')[1][0]) # int(sys.argv[2])
	# sublexes = ['sublex_%i' % k for k in range(6) if k!=sublex_id]
	# df = df.rename(columns={'sublex_%i' % sublex_id:'prob', 'MAP_sublex':'MAP'})
	# df = df.drop(labels= sublexes + [
	# 						# 'word_id',
	# 						'most_probable_sublexicon',
	# 						'customer_id',
	# 						# 'sub_orthography',
	# 						'actual_sublex',
	# 						'katakana',
	# 						# 'MAP'
	# 						], axis=1)
	# df['prob'] = df.prob.map(lambda value: '%0.6f' % value)
	# df.loc[:,'IPA'] = dummy_code_special_symbols(df.IPA)


	# Rendaku exception
	# sublexes = ['sublex_%i' % k for k in range(6)]
	# df['prob_to_sublex_5'] = df.sublex_5
	# df['orthography'] = df.lemma
	# df = df.drop(
	# 			sublexes
	# 			+
	# 			[
	# 				'lemma',
	# 				'lForm',
	# 				'wType',
	# 				'core_frequency',
	# 				'rendaku_rate_waei',
	# 				'customer_id'
	# 			]
	# 			,
	# 			axis=1
	# 			)
	# sublex_id = '5'
	# df['IPA'] = dummy_code_special_symbols(df.IPA)

	latex_table = df.to_latex(encoding='utf-8', index=False)
	latex_table = utf8_symbols_to_latex_commands(latex_table, sublex_id=sublex_id)
	print_utf8(latex_table)
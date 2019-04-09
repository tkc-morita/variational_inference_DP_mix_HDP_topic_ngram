# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os.path

penultimate_in_prefix = [u'n',u'j',u'ɾ']
last_in_prefix = [u'i',u'a',u'o']
consonants = [u'pːː',u'p',u'b',u'bːː',u'd',u'dːː']
vowels = [u'a',u'aː']

def barplot_log_probs(df_ngram, result_dir):
	ymin = np.floor(df_ngram.log_prob.min())
	for (prefix,c_type), sub_df in df_ngram.groupby(['prefix','c_type']):
		sub_df.loc[:,'target_syllable'] = pd.Categorical(
											sub_df.target_syllable,
											categories=[
												c+v+'END'
												for c in consonants
												for v in vowels
												if c_type in c
											])
		sns.barplot(
			x='sublex_id',
			hue='target_syllable',
			y='log_prob',
			data=sub_df,
			ci=None
		)
		plt.title(u'p(%s(ːː)a(ː)END | %s)' % (c_type,prefix))
		plt.ylim(ymin=ymin)
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig(os.path.join(result_dir, 'prob_a_prefix-%s_c-type-%s.png' % (prefix,c_type)), bbox_inches='tight')
		plt.clf()

if __name__ == '__main__':
	result_path = sys.argv[1]
	result_dir, filename = os.path.split(result_path)
	
	df_ngram = pd.read_csv(result_path, encoding='utf-8')
	df_ngram.loc[:,'decoded_value'] = df_ngram.decoded_value.str.replace(u'ä','a')
	df_context = df_ngram.decoded_context.str.replace(u'ä','a').str.replace(u'r',u'ɾ').str.split('_', expand=True).rename(columns={0:'c1', 1:'c2'})
	df_ngram = pd.concat([df_ngram,df_context], axis=1)

	
	df_ngram_c = df_ngram[(df_ngram.c1.isin(penultimate_in_prefix)) & (df_ngram.c2.isin(last_in_prefix)) & (df_ngram.decoded_value.isin(consonants))].rename(columns={'c1':'penult_in_prefix','c2':'last_in_prefix','decoded_value':'target_c','prob':'c_prob'})
	df_ngram_v = df_ngram[(df_ngram.c1.isin(last_in_prefix)) & (df_ngram.c2.isin(consonants)) & (df_ngram.decoded_value.isin(vowels))].rename(columns={'c1':'last_in_prefix','c2':'target_c','decoded_value':'target_v','prob':'v_prob'})
	df_ngram_v.loc[:,'penult_in_prefix'] = df_ngram_v.last_in_prefix.map(dict(zip(last_in_prefix,penultimate_in_prefix)))
	df_ngram_END = df_ngram[(df_ngram.c1.isin(consonants)) & (df_ngram.c2.isin(vowels)) & (df_ngram.decoded_value == u'END')].rename(columns={'c1':'target_c','c2':'target_v','decoded_value':'END','prob':'END_prob'})
	df_ngram_cv = pd.merge(df_ngram_c, df_ngram_v, on=['penult_in_prefix','last_in_prefix','target_c','sublex_id'])
	df_ngram_cvEND = pd.merge(df_ngram_cv, df_ngram_END, on=['target_c','target_v','sublex_id'])
	df_ngram_cvEND = df_ngram_cvEND.sort_values(
					[
						'penult_in_prefix',
						'last_in_prefix',
						'target_c',
						'target_v',
						'sublex_id'
						]
						)
	df_ngram_cvEND.loc[:,'target_syllable'] = df_ngram_cvEND.target_c + df_ngram_cvEND.target_v + 'END'
	df_ngram_cvEND.loc[:,'prefix'] = df_ngram_cvEND.penult_in_prefix + df_ngram_cvEND.last_in_prefix
	df_ngram_cvEND.loc[:,'c_type'] = df_ngram_cvEND.target_c.map(lambda x: x[0])
	df_ngram_cvEND.loc[:,'log_prob'] = df_ngram_cvEND.c_prob.apply(np.log) + df_ngram_cvEND.v_prob.apply(np.log) + df_ngram_cvEND.END_prob.apply(np.log)
	# print df_ngram_cvEND
	# df_code = pd.read_csv(os.path.join(result_dir, 'symbol_coding.csv'), encoding='utf-8')
	# df_code.set_index('code', inplace=True)
	# decoder = df_code.symbol.to_dict()

	# df_ngram['decoded_value'] = df_ngram.value.map(decoder)
	# df_ngram['decoded_context'] = df_ngram.context.map(lambda context: '_'.join(map(lambda code: decoder[int(code)], context.split('_'))))


	# df_ngram.to_csv(os.path.join(result_dir, 'Gelbart2005_trigram_probs.csv'), index=False, encoding='utf-8')
	barplot_log_probs(df_ngram_cvEND, result_dir)





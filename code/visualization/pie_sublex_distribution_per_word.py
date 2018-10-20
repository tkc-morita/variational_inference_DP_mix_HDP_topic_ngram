# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append('../data')
import my_autopct

def pie_sublex_distribution_per_word(df, result_path):
	df_sublex = df.loc[:,[column for column in df.columns if column.startswith('sublex_')]]
	for (col_name,row),ipa in zip(df_sublex.iterrows(), df.IPA):
		row.plot.pie(autopct=my_autopct.my_autopct, legend=True)
		plt.ylabel('')
		plt.title(u'Posterior predictive sublexical assignment probability of %s' % (ipa))
		plt.savefig(os.path.join(result_path, 'pie_%s.png' % ipa), bbox_inches='tight')
		plt.gcf().clear()


def bar_log_sublex_prob_ratio(df, result_path, target_sublex = 'sublex_3', group_id_name = 'group_identifier'):
	data = []
	for group_id, df_group in df.groupby(group_id_name):
		Foreign_word = df_group.loc[df_group.actual_sublex=='Foreign', 'IPA'].values[0]
		Native_word = df_group.loc[df_group.actual_sublex=='Native', 'IPA'].values[0]
		log_prob_Foreign_to_target_sublex = np.log(df_group.loc[df_group.actual_sublex=='Foreign', target_sublex].values[0])
		log_prob_Native_to_target_sublex = np.log(df_group.loc[df_group.actual_sublex=='Native', target_sublex].values[0])
		log_ratio = log_prob_Foreign_to_target_sublex - log_prob_Native_to_target_sublex
		data.append([Foreign_word+'-'+Native_word, - log_ratio],)
	
	df_data = pd.DataFrame(data, columns=['word_pair', 'log_sublex_prob_ratio'])
	# ordered_categories = [u'saɾada-haɾada', u'kɯɾabɯ-naɾabɯ', u'nabi-tabi', u'medaɾɯ-kɯdaɾɯ', u'maɡɯ-toɡɯ', u'neɡa-saɡa']#, u'nasa-mosa', u'sahaɾa-nohaɾa']
	# ordered_categories = [u'nasa-mosa', u'sahaɾa-nohaɾa']
	# ordered_categories = [u'saɾadːːa-haɾadːːa', u'kɯɾabːːɯ-naɾabːːɯ', u'nabːːi-tabːːi', u'medːːaɾɯ-kɯdːːaɾɯ', u'maɡːːɯ-toɡːːɯ', u'neɡːːa-saɡːːa']#, u'nasa-mosa', u'sahaɾa-nohaɾa']
	ordered_categories = [u'nasaː-mosaː', u'sahaɾaː-nohaɾaː']
	df_data.loc[:,'word_pair'] = pd.Categorical(df_data.word_pair,
									categories=ordered_categories
									)
	sns.set_style("whitegrid")
	ax = sns.barplot(x='word_pair', y='log_sublex_prob_ratio', data=df_data)
	plt.setp(ax.get_xticklabels(), rotation=45)
	# plt.ylim((-1,1))
	plt.tight_layout()
	plt.savefig(os.path.join(result_path, 'log_ratio_Native-Foreign_to-%s_V_target.png' % target_sublex), bbox_inches='tight')
	df_data.to_csv(os.path.join(result_path, 'log_ratio_Native-Foreign_to-%s_V_target.csv' % target_sublex), index=False, encoding='utf-8')




	
if __name__ == '__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, sep=',', encoding='utf-8')
	df['IPA'] = df.IPA.str.replace(u'ä',u'a').str.replace(u'r',u'ɾ').str.replace(u'g',u'ɡ')
	# df = df[df.experimenter == 'GelbartKawahara2007']

	result_path = os.path.splitext(data_path)[0]
	if not os.path.isdir(result_path):
		os.makedirs(result_path)
	
	# pie_sublex_distribution_per_word(df, result_path)

	sub_df = df[(df.experimenter == 'GelbartKawahara2007') & (df.stimulus_type == 'target') & (df.group_identifier > 5)]
	bar_log_sublex_prob_ratio(sub_df, result_path, target_sublex='sublex_4')
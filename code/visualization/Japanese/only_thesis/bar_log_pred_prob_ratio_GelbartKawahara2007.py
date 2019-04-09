# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append('../data')
import my_autopct

# def pie_sublex_distribution_per_word(df, result_path):
# 	df_sublex = df.loc[:,[column for column in df.columns if column.startswith('sublex_')]]
# 	for (col_name,row),ipa in zip(df_sublex.iterrows(), df.IPA):
# 		row.plot.pie(autopct=my_autopct.my_autopct, legend=True)
# 		plt.ylabel('')
# 		plt.title(u'Posterior predictive sublexical assignment probability of %s' % (ipa))
# 		plt.savefig(os.path.join(result_path, 'pie_%s.png' % ipa))
# 		plt.gcf().clear()


def bar_log_sublex_prob_ratio(df, result_path, log_prob = 'target_log_prob', group_id_name = 'group_identifier'):
	data = []
	for group_id, df_group in df.groupby(group_id_name):
		Foreign_word = df_group.loc[df_group.actual_sublex=='Foreign', 'IPA'].values[0]
		Native_word = df_group.loc[df_group.actual_sublex=='Native', 'IPA'].values[0]
		log_prob_Foreign_to_log_prob = (df_group.loc[df_group.actual_sublex=='Foreign', log_prob].values[0])
		log_prob_Native_to_log_prob = (df_group.loc[df_group.actual_sublex=='Native', log_prob].values[0])
		log_ratio = log_prob_Foreign_to_log_prob - log_prob_Native_to_log_prob
		data.append([Foreign_word+'-'+Native_word, log_ratio])
	
	df_data = pd.DataFrame(data, columns=['word_pair', 'log_pred_prob_ratio'])
	ordered_categories = [u'saɾada-haɾada', u'kɯɾabɯ-naɾabɯ', u'nabi-tabi', u'medaɾɯ-kɯdaɾɯ', u'maɡɯ-toɡɯ', u'neɡa-saɡa']
	# ordered_categories = [u'nasa-mosa', u'sahaɾa-nohaɾa']
	df_data.loc[:,'word_pair'] = pd.Categorical(df_data.word_pair,
									categories=ordered_categories
									)
	sns.set_style("whitegrid")
	ax = sns.barplot(x='word_pair', y='log_pred_prob_ratio', data=df_data)
	plt.setp(ax.get_xticklabels(), rotation=45)
	plt.ylim((-2.0, 2.0))
	plt.tight_layout()
	plt.savefig(os.path.join(result_path, 'log_ratio_Native-Foreign_%s_C.png' % log_prob))
	df_data.to_csv(os.path.join(result_path, 'log_ratio_Native-Foreign_%s_C.csv' % log_prob), index=False, encoding='utf-8')




	
if __name__ == '__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	df.loc[:,'control_word'] = df.prefix + df.control + df.suffix.fillna('')
	df.loc[:,'IPA'] = df.control_word.str.replace(u'ä',u'a').str.replace(u'r',u'ɾ').str.replace(u'g',u'ɡ').str.replace(',','')
	# df = df[df.experimenter == 'GelbartKawahara2007']

	result_path = os.path.splitext(data_path)[0]
	if not os.path.isdir(result_path):
		os.makedirs(result_path)
	
	# pie_sublex_distribution_per_word(df, result_path)

	sub_df = df[(df.experimenter == 'GelbartKawahara2007') & (df.group_identifier <= 5)]
	# sub_df.loc[:,'log_target-log_control'] = sub_df.target_log_prob - sub_df.control_log_prob
	log_prob = 'log_post_pred_prob_ratio_target_over_control'
	bar_log_sublex_prob_ratio(sub_df, result_path, log_prob=log_prob)
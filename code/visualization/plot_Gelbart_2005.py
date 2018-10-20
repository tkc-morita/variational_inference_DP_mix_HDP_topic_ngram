# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def barplot_prob_sum(df, x = None, y = 'normalized_prob_target', hue = None):
	ax = sns.barplot(
			x=x,
			y=y,
			hue=hue,
			data=df
			)
	# prob_sum = df.groupby(['c_length','v_length']).sum().reset_index()
	# print prob_sum
	# sns.barplot(
	# 	x=['c_length','v_length']
	# 	,
	# 	y='normalized_prob_target'
	# 	,
	# 	data=df
	# )
	# plt.setp(ax.get_xticklabels(), rotation=45)
	# plt.ylabel('Normalized posterior predictive probability')
	# plt.ylabel('Normalized posterior predictive probability of [a:]')
	# plt.ylabel('Expected proportion of [a:]-responses')
	plt.ylabel('Expected counts of responses')
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	data_path = sys.argv[1]

	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	df['c_type'] = df['target_c'].map(lambda x: x[0]+u'(ːː)')
	df['prefix'] = df.prefix.str.replace(',','')
	df['c_length'] = pd.Categorical(df.c_length, categories=['short','long'])
	df['v_length'] = pd.Categorical(df.v_length, categories=['short','long'])

	# Each stimulus was responded at least by 6 subjects.
	df.loc[:,'expected_response_count'] = df.normalized_prob_target * 6 * 8 * 6

	# keni was responded twice as many as others.
	df.loc[
		df.prefix==u'keni',
		'expected_response_count'
		] *= 2




	x = 'c_length'
	# x = 'target_c'
	# x = 'prefix'
	# x = 'c_type'

	hue = 'v_length'
	# hue = 'c_length'

	# target = 'c_length'
	# target = 'v_length'

	# df = df[df.c_length=='long']
	sub_df = df.groupby([x,hue]).expected_response_count.sum().to_frame().reset_index().rename(columns={0:'expected_response_count'})
	# sub_df = df.groupby([x,hue,target]).expected_response_count.sum().to_frame().reset_index().rename(columns={0:'expected_response_count'})
	sub_df.loc[:,'expected_response_proportion'] = sub_df.expected_response_count / sub_df.expected_response_count.sum()
	# for (x_val,hue_val),subsub_df in sub_df.groupby([x,hue]):
	# 	sub_df.loc[
	# 		(sub_df[x]==x_val)
	# 		&
	# 		(sub_df[hue]==hue_val)
	# 		,
	# 		'expected_response_proportion'
	# 	] = subsub_df.expected_response_count / subsub_df.expected_response_count.sum()
	# sub_df = sub_df[sub_df[target]=='long']


	barplot_prob_sum(sub_df, x=x, hue=hue, y='expected_response_count')
	# barplot_prob_sum(sub_df, x=x, hue=hue, y='expected_response_proportion')

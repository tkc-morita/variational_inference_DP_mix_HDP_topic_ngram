# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os.path


def summarize_by(df, target_var, groups): #x, y, hue=None):
	# df.groupby(groups)[target_var].sum().plot.bar()
	x = groups[0]
	hue = groups[1]
	y = target_var
	sns.barplot(
				x=x,
				y=y,
				hue=hue,
				data=df,
				ci=None
			)
	plt.tight_layout()
	plt.show()

def get_entropy(df, tokens={u'ʨ,i,r,o':1, u'k,e,n,i':1, u's,ɯ,j,ä':1}):
	rows = []
	groups = ['c_type','v_length','trial_name','prefix']
	for (c_type,v_length,trial_name,prefix),sub_df in df.groupby(groups):
		normalized_prob = sub_df.normalized_prob_target
		normalized_prob /= np.sum(normalized_prob)
		entropy = - np.sum(normalized_prob * np.log(normalized_prob))
		rows += [(c_type,v_length,trial_name,entropy,)] * tokens[prefix]
	return pd.DataFrame(
		rows,
		columns=groups[:-1]+['entropy']
	)

def get_surprisal(df, tokens={u'ʨ,i,r,o':1, u'k,e,n,i':1, u's,ɯ,j,ä':1}):
	rows = []
	groups = ['c_type','v_length','trial_name','prefix']
	for (c_type,v_length,trial_name,prefix),sub_df in df.groupby(groups):
		print sub_df
		normalized_prob = sub_df.normalized_prob_target
		normalized_prob /= np.sum(normalized_prob)
		if c_type == 'p':
			foreign_length = 'short'
			target_c = c_type
		else:
			foreign_length = 'long'
			target_c = c_type + u'ːː'
		# print 'Foreign', sub_df[sub_df.c_length==foreign_length]
		# print 'non-Foreign', sub_df[sub_df.c_length!=foreign_length]
		surprisal = - np.log(normalized_prob[sub_df.c_length==foreign_length].values[0])
		rows += [(c_type,v_length,trial_name,target_c,surprisal,)] * tokens[prefix]
	return pd.DataFrame(
		rows,
		columns=groups[:-1]+['target_c','surprisal']
	)

if __name__ == '__main__':
	data_path = sys.argv[1]
	result_dir = os.path.split(data_path)[0]

	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
	df['c_type'] = df.target_c.map(lambda x: x[0])
	df['prefix'] = pd.Categorical(df.prefix, categories=[u'ʨ,i,r,o', u'k,e,n,i', u's,ɯ,j,ä'], ordered=True)
	df['c_length'] = pd.Categorical(df.c_length, categories=['short','long'])
	df['v_length'] = pd.Categorical(df.v_length, categories=['short','long'])
	# df = df[df.c_length=='long']
	# df = df[df.prefix == u'ʨ,i,r,o']
	
	# y = "normalized_prob_target"
	y = 'entropy'
	# y = 'surprisal'
	
	# groups = ["c_length", "v_length"]
	# groups = ["c_length","v_length", "c_type"]
	# groups = ["v_length", "prefix"]


	# tokens={u'ʨ,i,r,o':6, u'k,e,n,i':12, u's,ɯ,j,ä':6}

	# Entropy
	groups = ["c_type", "v_length"]
	df_entropy = get_entropy(df)#, tokens=tokens)


	# Surprisal
	# groups = ["target_c", "v_length"]
	# df_surprisal = get_surprisal(df)#, tokens=tokens)
	# df_surprisal.to_csv(os.path.join(result_dir,'surprisal.csv'), index=False, encoding='utf-8')

	# summarize_by(df, y, groups)
	summarize_by(df_entropy, y, groups)
	# summarize_by(df_surprisal, y, groups)
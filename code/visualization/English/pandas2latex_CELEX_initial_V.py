import pandas as pd
import numpy as np
import pandas2latex_CELEX as p2l
import sys


def formatter_counts(x):
	return ('%.2f' % x)

def formatter_percent(x):
	return (r'%.2f\%%' % x)

def format_sublex_name(sublex_name):
	return (r'\textsc{Sublex}\textsubscript{$\approx$%s}' % sublex_name)
	# return (r'\textsc{Sublex}\textsubscript{%s}' % sublex_name)

def rename_sublex(sublex_name):
	ix = int(sublex_name)
	ix2name = {0:'-ability', 2:'Latinate', 5:'Germanic'}
	return ix2name[ix]


if __name__ == '__main__':
	# pd.set_option('display.max_colwidth', -1)

	path = sys.argv[1]
	df = pd.read_csv(path, encoding='utf-8')

	# df.loc[:,'value'] = df.value.map(p2l.disc2latex_func)

	df = df[df.sublex.isin([2,5])]
	# df_formatted.loc[:,'sublex'] = df.sublex.map(rename_sublex).map(format_sublex_name)
	df_formatted = pd.pivot_table(df, values='representativeness', index='vowel', columns = 'sublex')



	df_formatted = df_formatted.sort_values(2, ascending=False)

	df_formatted = df_formatted.set_index(df_formatted.index.map(p2l.disc2latex_func))

	df_formatted = df_formatted.rename(columns={ix:format_sublex_name(rename_sublex(ix)) for ix in [2,5]})
	
	latex_table = df_formatted.to_latex(
					encoding='utf-8',
					escape = False,
					longtable = False,
					# index = False,
					# formatters = [lambda x: '%i' % x, formatter_counts, formatter_percent, formatter_counts, formatter_percent, formatter_counts, formatter_percent]
					)
	with open(sys.argv[2], 'w') as f:
		f.write(latex_table)




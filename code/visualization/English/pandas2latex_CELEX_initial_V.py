import pandas as pd
import numpy as np
import pandas2latex_CELEX as p2l
import pytablewriter, sys


def formatter_counts(x):
	return ('%.2f' % x)

def formatter_percent(x):
	return (r'%.2f\%%' % x)


if __name__ == '__main__':
	# pd.set_option('display.max_colwidth', -1)

	path = sys.argv[1]
	df = pd.read_csv(path, encoding='utf-8')

	# df.loc[:,'value'] = df.value.map(p2l.disc2latex_func)

	df_formatted = pd.pivot_table(df, values='expected_frequency', index='value', columns = 'sublex_id')

	frequency = df.groupby(['value']).expected_frequency.sum()
	df_formatted['total'] = frequency.map(np.round)


	for col in df_formatted.columns:
		if isinstance(col, int):
			df_formatted.loc[:,str(col)+'_percent'] = df_formatted.loc[:,col] / df_formatted.total * 100

	df_formatted = df_formatted.sort_values('5_percent', ascending=False)

	df_formatted = df_formatted.loc[:,[
								'total',
								0,
								'0_percent',
								2,
								'2_percent',
								5,
								'5_percent'
							]]

	df_formatted = df_formatted.set_index(df_formatted.index.map(p2l.disc2latex_func))
	
	latex_table = df_formatted.to_latex(
					encoding='utf-8',
					escape = False,
					longtable = False,
					# index = False,
					formatters = [lambda x: '%i' % x, formatter_counts, formatter_percent, formatter_counts, formatter_percent, formatter_counts, formatter_percent]
					)
	with open(sys.argv[2], 'w') as f:
		f.write(latex_table)




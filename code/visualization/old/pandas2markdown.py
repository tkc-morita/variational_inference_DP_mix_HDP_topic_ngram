# coding: utf-8

import pandas as pd
import pytablewriter, sys

def pandas2markdown(df):
	writer = pytablewriter.MarkdownTableWriter()
	writer.from_dataframe(df)
	writer.write_table()

if __name__ == '__main__':
	path = sys.argv[1]
	df = pd.read_csv(path, encoding='utf-8')

	df = df.drop(labels='sublex_id', axis=1)

	# sublex_id = int(sys.argv[2])
	# sublexes = ['sublex_%i' % k for k in range(6) if k!=sublex_id]
	# df = df.rename(columns={'sublex_%i' % sublex_id:'representativeness'})
	# df = df.drop(labels= sublexes + ['word_id','sub_orthography','actual_sublex'], axis=1)

	pandas2markdown(df)
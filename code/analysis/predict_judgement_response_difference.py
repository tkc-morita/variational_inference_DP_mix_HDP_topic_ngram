# coding: utf-8

import numpy as np
import pandas as pd
import sys, os.path
sys.path.append('/Users/takashi/Dropbox (Personal)/pyworks/poibin')
import poibin
# import poisson_binomial as poibin


def get_poisson_binomial_for_word_pair(num_responses, p1, p2):
	ps = ([p1] * num_responses) + ([1 - p2] * num_responses)
	# return poibin.Poisson_Binomial(ps)
	return poibin.PoiBin(ps)


def main_loop(p1s, p2s, num_responses_list, identifiers):
	table = []
	for p1, p2, num_responses, identifier in zip(p1s,p2s,num_responses_list, identifiers):
		pb = get_poisson_binomial_for_word_pair(num_responses, p1, p2)
		for k in range(num_responses+1):
			table.append([identifier,p1,p2,k,pb.pmf(k),num_responses])
	return pd.DataFrame(table, columns=['identifier','p1','p2','k','prob','num_responses'])


if __name__ == '__main__':
	data_path = sys.argv[1]
	# num_responses = sys.argv[2]

	df = pd.read_csv(data_path, sep='\t', encoding='utf-8')

	df_Native = df[df.actual_sublex == 'Native'].rename(columns={'normalized_control_prob':'p2s', 'control_word':'Native_word'})
	df_Foreign = df[df.actual_sublex == 'Foreign'].rename(columns={'normalized_target_prob':'p1s', 'control_word':'Foreign_word', 'num_responses':'hoge'})

	df_focus = pd.merge(df_Native, df_Foreign, on='group_identifier')
	df_focus['identifier'] = df_focus.Foreign_word.str.replace(',','') + '-' + df_focus.Native_word.str.replace(',','')
	
	result_root = os.path.splitext(data_path)[0]

	df_result = main_loop(df_focus.p1s, df_focus.p2s, df_focus.num_responses.astype(int), df_focus.identifier)
	df_result['response_difference'] = df_result.k - df_result.num_responses
	# df_result['Native_word'] = df_focus.Native_word
	# df_result['Foreign_word'] = df_focus.Foreign_word
	df_result.to_csv(result_root+'_response-to-long-difference-from-Foreign-Native.tsv', sep='\t', encoding='utf-8', index=False)

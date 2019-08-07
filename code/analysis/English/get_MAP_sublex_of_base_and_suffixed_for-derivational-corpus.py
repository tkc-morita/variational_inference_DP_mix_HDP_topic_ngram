# coding: utf-8

import pandas as pd
import argparse, os.path


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('morph_path', type=str, help='Path to the csv with morphological info.')
	# parser.add_argument('phone_path', type=str, help='Path to the tsv with phonological info.')
	parser.add_argument('classification_path', type=str, help='Path to the classification results.')
	parser.add_argument('--top_k', type=int, default=None, help='If specified, limit to the top k type-frequent suffixes.')
	args = parser.parse_args()

	df_data = pd.read_csv(args.morph_path, sep=',', encoding='utf-8')
	# df_phone = pd.read_csv(args.phone_path, sep='\t', encoding='utf-8')
	# df_data['DISC'] = df_phone.DISC
	df_class = pd.read_csv(args.classification_path, encoding='utf-8')
	df_data['map_sublex_suffixed'] = df_class.most_probable_sublexicon

	df_suffixed = df_data[df_data.affix.fillna('').str.startswith('-')]
	if not args.top_k is None:
		frequent_suffixes = df_suffixed.affix.value_counts().index[:args.top_k]
		print(frequent_suffixes)
		df_suffixed  = df_suffixed[df_suffixed.affix.isin(frequent_suffixes)]
	df_suffixed = pd.merge(df_suffixed.rename(columns={'Head':'lemma'}), df_data.loc[:,['Head','category','map_sublex_suffixed']].rename(columns={'map_sublex_suffixed':'map_sublex_base'}), how='left', left_on=['base','base_category'], right_on=['Head','category'])
	df_suffixed = df_suffixed.drop_duplicates(subset=['IdNum','map_sublex_base'])
	print('Disagreement in base category:')
	disagreed_base_category = df_suffixed.duplicated(subset=['IdNum'], keep=False)
	print(df_suffixed[disagreed_base_category])
	df_suffixed = df_suffixed[~disagreed_base_category]
	df_suffixed = df_suffixed[~df_suffixed.map_sublex_base.isnull()]

	save_dir = os.path.dirname(args.classification_path)
	filename = os.path.splitext(os.path.basename(args.morph_path))[0] + '.csv'
	df_suffixed.to_csv(os.path.join(save_dir, filename), index=False)
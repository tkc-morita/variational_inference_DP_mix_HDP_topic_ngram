# coding: utf-8

import pandas as pd
import argparse, os.path

def get_map_sublex(id_nums, df_class):
	id_nums = [int(ix) for ix in id_nums.split(';')]
	map_classes = df_class.loc[df_class.IdNum.isin(id_nums),'most_probable_sublexicon'].unique().tolist()
	return ';'.join(sorted(map_classes))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('morph_path', type=str, help='Path to the csv with morphological info.')
	parser.add_argument('classification_path', type=str, help='Path to the classification results.')
	parser.add_argument('phone_path', type=str, help='Path to the tsv with phonological info.')
	args = parser.parse_args()

	df_data = pd.read_csv(args.morph_path, sep=',', encoding='utf-8')
	df_suffixed = df_data[~df_data.base_IdNum.isnull()]
	df_phone = pd.read_csv(args.phone_path, sep='\t', encoding='utf-8')
	df_class = pd.read_csv(args.classification_path, encoding='utf-8')
	df_class['IdNum'] = df_phone['rank']
	df_suffixed['map_sublex_base'] = df_suffixed.base_IdNum.map(lambda x: get_map_sublex(x, df_class))

	df_suffixed = df_suffixed[~df_suffixed.map_sublex_base.isnull()]
	print('Disagreement in the MAP sublex:')
	disagreed_base_classification = df_suffixed.map_sublex_base.str.contains(';')
	print(df_suffixed[disagreed_base_classification])
	df_suffixed = df_suffixed[~disagreed_base_classification]

	save_dir = os.path.dirname(args.classification_path)
	filename = os.path.splitext(os.path.basename(args.morph_path))[0] + '.csv'
	df_suffixed.to_csv(os.path.join(save_dir, filename), index=False)
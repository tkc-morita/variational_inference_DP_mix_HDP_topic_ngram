# coding: utf-8

import pandas as pd

import kana2ipa
import get_core_nouns

import argparse, os.path


def parse_argments():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the BCCWJ word frequency list. The file must be pre-converted into a utf-8 tsv file.')
	parser.add_argument('--save', type=str, help='Path to the formatted file. "_with-IPA-of-core-nouns" is attached to the original if unspecified.', default=None)
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_argments()

	df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')

	df_core = get_core_nouns.get_core_nouns(df)
	df_core['IPA_csv'] = df_core.lForm.apply(kana2ipa.kana2ipa)

	if args.save is None:
		directory, data_filename = os.path.split(args.data_path)
		save_filename_wo_ext, ext = os.path.splitext(data_filename)
		save_filename_wo_ext += '_with-IPA-of-core-nouns'
		save_path = os.path.join(directory, save_filename + ext)
	else:
		save_path = args.save
	df_core.to_csv(save_path, index=False, encoding='utf-8', sep='\t')
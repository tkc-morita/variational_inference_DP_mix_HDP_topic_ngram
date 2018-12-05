# coding: utf-8

import pandas as pd
import itertools, sys


def get_core_nouns(df):
	"""
	Extract nouns with non-zero core_frequency from the word frequency list of BCCWJ.
	"""
	df = df[df.pos.str.startswith(u'名詞')
				& (df.core_frequency>0)
				& (~df.lForm.str.contains(u'■')) # Words including personal info are masked by ■, and cannot be used.
				]
	return df
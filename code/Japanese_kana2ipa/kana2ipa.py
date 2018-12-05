#coding: utf-8

import numpy as np
import pandas as pd
import romkan
import sys, os.path


def kana2ipa(kana_word):
	"""
	Translate a word (utf string) written in kana into IPA(-ish).
	Basically following Vance (2008).
	Some non-IPA transcriptions below (only those that could appear in the output):
	- 9: [ɰ̃]. I avoided complex symbols consisting of multiple utf characters.
	- ä: Just [a]. Intended to mean fronted [ɑ], which Vance (2008) transcribe as "ɑ" above "+" in his narrow transcription and [a] elsewhere.
	- r: [ɾ]. Just forgot to use the correct one.
	- ːː (two successive ː): Superlong. Used by Vance (2008) for geminates.
	- ?: Unkmnown symbol. Used to encode ■ in the BCCWJ.
	"""
	kana_word = '$'+kana_word\
				.replace(u'■',u'?')\
				.replace(u'・',u'')\
				.replace(u'ツァ',u'ʦa')\
				.replace(u'ツィ',u'ʦi')\
				.replace(u'ツェ',u'ʦe')\
				.replace(u'ツォ',u'ʦo')\
				.replace(u'トゥ',u'tu')\
				.replace(u'ヂ',u'ジ')\
				.replace(u'ヅ',u'ズ')\
				.replace(u'ッ',u'2')\
				.replace(u'ウォ',u'wo')\
				.replace(u'ン',u'9ː')\
				.replace(u'オオウ',u'オーウ') + '#'	#End symbol.
	segmented_word = romkan.to_roma(kana_word)\
					.replace('ts',u'ʦ')\
					.replace('ch',u'ʨ')\
					.replace('sh',u'ɕ')\
					.replace('f',u'ɸ')\
					.replace('j',u'ʥ')\
					.replace('y','j')\
					.replace('hi',u'çi')\
					.replace('hj',u'çj')\
					.replace('$z',u'$ʣ')\
					.replace(u'9ːz',u'9ːʣ')\
					.replace('2z',u'2ʣ')\
					.replace('ei9',u'eI9').replace('ou9',u'oU9')\
					.replace('ei',u'eː').replace('ou',u'oː')\
					.replace('aa',u'aː').replace('ii',u'iː').replace('uu',u'uː').replace('ee',u'eː').replace('oo',u'oː')\
					.replace('nj',u'ɲj').replace('ni',u'ɲi')\
					.replace('pj','p^j').replace('bj','b^j').replace('mj','m^j')\
					.replace('tj','t^j').replace('dj','d^j')\
					.replace('kj','k^j').replace('gj','g^j')\
					.replace(u'ɸj',u'ɸ^j').replace('rj',u'r^j')\
					.replace('pi','p^i').replace('bi','b^i').replace('mi','m^i')\
					.replace('ti','t^i').replace('di','d^i')\
					.replace('ki','k^i').replace('gi','g^i')\
					.replace('ri','r^i')\
					.replace(u'9ːt',u'nːt').replace(u'9ːd',u'nːd').replace(u'9ːn',u'nːː')\
					.replace(u'9ːz',u'nːz').replace(u'9ːr',u'nːr').replace(u'9ːʦ',u'nːʦ')\
					.replace(u'9ːp',u'mːp').replace(u'9ːb',u'mːb').replace(u'9ːm',u'mːː')\
					.replace(u'9ːk',u'ŋːk').replace(u'9ːg',u'ŋːg')\
					.replace(u'9ːʨ',u'ɲːʨ').replace(u'9ːʥ',u'ɲːʥ')\
					.replace(u'9ːɲ',u'ɲːː')\
					.replace(u'9ː#',u'ɴː#')\
					.replace('-',u'ː')\
					.replace('x','')\
					.replace('a',u'ä').replace('u',u'ɯ')\
					.replace('#','').replace('$','')
	segmented_word = segmented_word.replace(u'I',u'i').replace(u'U',u'ɯ').replace(u'iːː',u'iiː').replace(u'oːː',u'ooː')
	listed_segments = []
	geminate = False
	for segment in segmented_word:
		if geminate:
			listed_segments.append(segment)
			listed_segments.append(u'ː')
			listed_segments.append(u'ː')
			geminate=False
		elif segment==u'2':
			geminate=True
		else:
			listed_segments.append(segment)
	output = u''.join(listed_segments)
	output = output.replace('^','') # "^"" initially intended to denote palatalization superscript. But deprecated.
	output = segmentize_IPA(output)
	return output

def segmentize_IPA(ipa_word):
	"""
	Comma-separate IPA(-ish) symbols of a word.
	"""
	return ','.join(list(ipa_word)).replace(u',ː',u'ː')
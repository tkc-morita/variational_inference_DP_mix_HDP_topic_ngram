# coding: utf-8

import pandas as pd
# import pandas2latex as p2l
import argparse

disc2latex = {
	"g":r"{\textscriptg}",
	"N":r"\textipa{N}",
	"T":r"{\texttheta}",
	"D":r"\textipa{D}",
	"S":r"{\textesh}",
	"Z":r"{\textyogh}",
	"G":r"{\textgamma}",
	"+":r"pf",
	"=":r"{\texttslig}",
	"J":r"{\textteshlig}",
	"_":r"{\textdyoghlig}",
	"C":r"\textsyllabic{\textipa{N}}",
	"F":r"\textsyllabic{m}",
	"H":r"\textsyllabic{n}",
	"P":r"\textsyllabic{l}",
	"R":r"*",
	"i":r"i{\textlengthmark}",
	"!":r"i{\textlengthmark}{\textlengthmark}",
	"#":r"{\textscripta}{\textlengthmark}",
	"a":r"a{\textlengthmark}",
	"$":r"{\textopeno}{\textlengthmark}",
	"u":r"u{\textlengthmark}",
	"3":r"{\textrevepsilon}{\textlengthmark}",
	"1":r"e{\textsci}",
	"2":r"a{\textsci}",
	"4":r"{\textopeno}{\textsci}",
	"5":r"{\textschwa}{\textupsilon}",
	"6":r"a{\textupsilon}",
	"7":r"{\textsci}{\textschwa}",
	"8":r"{\textepsilon}{\textschwa}",
	"9":r"{\textupsilon}{\textschwa}",
	"K":r"{\textepsilon}{\textsci}",
	"M":r"{\textscripta}u",
	"I":r"{\textsci}",
	"Y":r"{\textscy}",
	"E":r"{\textepsilon}",
	"{":r"{\ae}",
	"&":r"a",
	"A":r"{\textscripta}",
	"Q":r"{\textturnscripta}",
	"V":r"{\textturnv}",
	"O":r"{\textopeno}",
	"U":r"{\textupsilon}",
	"@":r"{\textschwa}",
	"c":r"{\~{\ae}}",
	"q":r"{\~{\textscripta}}{\textlengthmark}",
	"0":r"{\~{\ae}}{\textlengthmark}",
	"~":r"{\~{\textturnscripta}}{\textlengthmark}",
	"END":r"\texttt{END}",
	"START":r"\texttt{START}"
}

def convert_context(context):
	if context.startswith('__'):
		context = context.replace('__', '_,')
	elif context.endswith('__'):
		context = context.replace('__', ',_')
	else:
		context = context.replace('_', ',')
	
	return ','.join([disc2latex_func(disc) for disc in context.split(',')])
	# return context

def disc2latex_func(disc):
	if disc in disc2latex:
		return disc2latex[disc]
	else:
		return disc

if __name__ == '__main__':
	pd.set_option('display.max_colwidth', -1)

	parser = argparse.ArgumentParser()
	parser.add_argument('table_path', type=str, help='Path to the table.')
	parser.add_argument('save_path', type=str, help='Path to the tex file to save the table.')
	parser.add_argument('--sublex_id', type=int, default=None, help='ID # of sublex. Used when the table_path does not include this info.')
	args = parser.parse_args()

	path = args.table_path
	df = pd.read_csv(path, encoding='utf-8', sep='\t')

	# For ngram rep/
	# sublex_id = int(path.split('sublex-')[1][0]) # int(sys.argv[2])
	# df.loc[:,'context'] = df.decoded_context.map(convert_context)
	# df.loc[:,'value'] = df.decoded_value.map(disc2latex_func)
	# representativeness_latex = r'$R(x_{\textrm{new}}, \mathbf{u},' + ' {sublex})$'.format(sublex=sublex_id)
	# df.loc[:,representativeness_latex] = df.representativeness
	# df.loc[:,'freq.'] = df.frequency
	# df = df[['context', 'value', representativeness_latex, 'freq.']]

	# For joint ngram rep
	df.loc[:,'IPA'] = df.substring_csv.map(lambda string: ','.join([disc2latex_func(code) for code in string.split(',')]))
	df.loc[:,'representativeness'] = df.representativeness.map(lambda value: '%0.6f' % value)
	df = df[['rank','IPA','representativeness']]


	# For probable words
	# # sublex_id = int(path.split('sublex-')[1][0]) # int(sys.argv[2])
	# sublex_id = args.sublex_id
	# # sublexes = ['sublex_%i' % k for k in range(6) if k!=sublex_id]
	# df = df.rename(columns={'sublex_%i' % sublex_id:'prob'})
	# df['prob'] = df.prob.map(lambda value: '%0.6f' % value)
	# df.loc[:,'IPA'] = df.DISC.map(lambda string: ''.join([disc2latex_func(code) for code in string]))
	# df = df[df.origin.isin(['Latin','French'])]

	# df['rank'] = df.index + 1

	# # df['origin'] = df.origin.str.replace('&',r'\&').str.replace('_',' ').str.replace('|','/').str.replace('IMITATIVE',r'\textsc{imitative}').str.replace('SYMBOLIC',r'\textsc{symbolic}').str.replace('UNKNOWN',r'\textsc{unknown}').str.replace('NO',r'\textsc{unwritten}')
	# # df = df[['orthography','IPA','origin','prob']]
	# # df = df[['orthography','IPA','prob']]
	# df = df[['lemma','IPA','origin','prob']]

	# df = df[['rank','orthography','IPA','prob']]

	# print df.family.value_counts()

	latex_table = df.to_latex(encoding='utf-8', index=False, escape = False, longtable = False)
	# latex_table = latex_table.replace(r'\textbackslash','\\').replace('\{','{').replace('\}','}')
	# p2l.print_utf8(latex_table)

	# latex_table = latex_table.replace(r'ation',r'{\color[rgb]{0.999996,0.999939,0.041033}ation}').replace(r'e{\textsci}{\textesh}\textsyllabic{n}',r'{\color[rgb]{0.999996,0.999939,0.041033}e{\textsci}{\textesh}\textsyllabic{n}}')
	# latex_table = latex_table.replace(r'ful',r'{\color[rgb]{0.999996,0.999939,0.041033}ful}').replace(r'f{\textupsilon}l',r'{\color[rgb]{0.999996,0.999939,0.041033}f{\textupsilon}l}')
	# latex_table = latex_table.replace(r'ness',r'{\color[rgb]{0.999996,0.999939,0.041033}ness}').replace(r'n{\textsci}s',r'{\color[rgb]{0.999996,0.999939,0.041033}n{\textsci}s}')
	# latex_table = latex_table.replace(r'ability',r'{\color[rgb]{0.999996,0.999939,0.041033}ability}').replace(r'{\textschwa}b{\textsci}l{\textschwa}t{\textsci}',r'{\color[rgb]{0.999996,0.999939,0.041033}{\textschwa}b{\textsci}l{\textschwa}t{\textsci}}')
	# latex_table = latex_table.replace(r'ibility',r'{\color[rgb]{0.999996,0.999939,0.041033}ibility}')

	with open(args.save_path, 'w') as f:
		f.write(latex_table)
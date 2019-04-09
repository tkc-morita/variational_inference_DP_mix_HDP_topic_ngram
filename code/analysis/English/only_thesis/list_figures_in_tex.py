# coding: utf-8

import os, sys

figure_start = """\\begin{figure}
	\\centering
"""

figure_end = """
	\\caption{
		(cont.)
		Posterior distribution of
		sublexical assignment of words created
		by affixations.
		The %i-%i-th most commonly used suffixes are listed.
	}
	\label{fig:English-pie-suffix-%i}
\\end{figure}
"""

subcapbox = """	\\subcaptionbox{
		\\emph{-%s}
		\\label{fig:English-pie-%s}
	}{
		\\includegraphics[width=0.45\\textwidth]{../figures/indep/CELEX-English/prefix-suffix/%s}
	}"""

def main_loop(directory):
	content = ''
	grouped_figures = figure_start
	group_ix = 0
	for ix, filename in enumerate([fn for fn in os.listdir(directory) if not fn.startswith('.') and 'suffix' in fn]):
		sub_ix = ix % 6
		if sub_ix == 5:
			grouped_figures += fill_out_template(filename) + '\n'
			grouped_figures += (figure_end % (ix - 4, ix + 1, group_ix)) + '%\n'
			content += grouped_figures
			group_ix += 1

			grouped_figures = figure_start
		else:
			grouped_figures += fill_out_template(filename)
			if sub_ix % 2:
				grouped_figures += '\n\n'
			else:
				grouped_figures += '%\n'
	return content

def fill_out_template(filename):
	_,affix_type,rank,affix_name = os.path.splitext(filename)[0].split('_')
	return (subcapbox % (affix_name, affix_type + '-' + affix_name, filename))


if __name__ == '__main__':
	directory = sys.argv[1]

	content = main_loop(directory)

	with open('English-prefix-suffix.tex', 'w') as f:
		f.write(content)


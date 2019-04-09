# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path, sys



def plot_prediction_against_data(df, prediction, data, annotation = None):
	ax = df.plot.scatter(prediction, data)

	xmin,xmax = ax.get_xlim()
	x_interval_over_10 = (xmax - xmin) / 10.0
	ymin,ymax = ax.get_ylim()
	y_interval_over_10 = (ymax - ymin) / 10.0

	if not annotation is None:
		for p, d, a in zip(df[prediction].tolist(), df[data].tolist(), annotation):
			p_label_pos = p + x_interval_over_10/8
			d_label_pos = d + y_interval_over_10/8
			if xmax < p + x_interval_over_10:
				p_label_pos -= x_interval_over_10*2.1
			if ymax < d + y_interval_over_10:
				d_label_pos -= y_interval_over_10/2
			ax.annotate(
				a,
				xy = (p, d),
				xytext = (p_label_pos,d_label_pos)
			)
	
	plt.show()



if __name__ == '__main__':
	datapath = sys.argv[1]
	result_dir,filename = os.path.split(datapath)
	fileroot,ext = os.path.splitext(filename)

	if ext == '.tsv':
		sep = '\t'
	else:
		sep = ','
	df = pd.read_csv(datapath, sep=sep, encoding='utf-8')

	
	# Focus on Moreton and Amano (1999).
	# df.loc[:,'prefix'] = df.prefix.str.replace('r',u'ɾ').str.replace(u'ä', r'a')
	# df = df[df.experimenter == 'MoretonAmano1999']
	# xcol = "C"
	# ycol = "C'"
	# zcol = "log posterior predictive probability ratio of [a] to [a:]"
	# df[xcol] = df.prefix.map(lambda ipa_csv: ipa_csv.replace(',','').split('o')[0])
	# df[ycol] = df.prefix.map(lambda ipa_csv: ipa_csv.replace(',','').split('o')[1])
	# df[zcol] = - df.log_post_pred_prob_ratio_target_over_control

	# ratio_min_dfmaxmin = df.experiment_result.min() / (df.experiment_result.max() - df.experiment_result.min())
	# bottom = df[zcol].min() - ratio_min_dfmaxmin * (df[zcol].max() - df[zcol].min())

	# # get_3dbar(df, ycol, xcol, zcol, bottom = bottom)

	# annotation = (df.prefix.str.replace(',','') + '_').tolist()
	# experiment = 'Duration threshold between [a] and [a:] (msec)'
	# df[experiment] = df.experiment_result
	# plot_prediction_against_data(df, zcol, experiment, annotation = annotation)


	# Focus on Gelbart & Kawahara (2007)
	experiment = 'Mean difference in # of "long" responses'
	df[experiment] = df.mean_response_diff
	predictor = 'log ratio of posterior predictive probability of classification into sublex_5'
	df[predictor] = df.log_sublex_prob_ratio
	# predictor = 'log ratio of posterior predictive probability ratio'
	# df[predictor] = df.log_pred_prob_ratio
	plot_prediction_against_data(df, predictor, experiment, annotation = df.word_pair)
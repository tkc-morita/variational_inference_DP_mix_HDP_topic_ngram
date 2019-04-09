# coding: utf-8

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path, sys

def get_3dbar(df, xcol, ycol, zcol, bottom = 0):
	datasize = df.shape[0]
	width = 0.8
	dx = [width] * datasize
	dy = [width] * datasize
	dz = df[zcol] - bottom

	x2pos = {value:pos for pos,value in enumerate(reversed(df[xcol].drop_duplicates().tolist()), start=1)}
	y2pos = {value:pos for pos,value in enumerate(df[ycol].drop_duplicates().tolist(), start=1)}
	
	xpos = df[xcol].map(x2pos)
	ypos = df[ycol].map(y2pos)
	zpos = [bottom] * datasize



	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')

	ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='y')

	ax1.set_xlabel(xcol)
	ax1.set_ylabel(ycol)
	ax1.set_zlabel(zcol)

	ax1.set_xticks(np.array([pos for value,pos in x2pos.items()]) + 0.5*width)
	ax1.set_xticklabels([value for value,pos in x2pos.items()])
	ax1.set_yticks(np.array([pos for value,pos in y2pos.items()]) + 0.5*width)
	ax1.set_yticklabels([value for value,pos in y2pos.items()])


	for x,y,z in zip(xpos.tolist(), ypos.tolist(), df[zcol].tolist()):
		ax1.text(
			x + 0.3 * width
			# x + 0.15 * width
			,
			y + 0.3 * width
			# y + 0.15 * width
			,
			z
			,
			'%.1f' % z
			# '%.6f' % z
		)

	plt.show()

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
				p_label_pos -= x_interval_over_10/2
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
	df = pd.read_csv(datapath, sep='\t', encoding='utf-8')

	df.loc[:,'prefix'] = df.prefix.str.replace('r',u'ɾ').str.replace(u'ä', r'a')
	
	# Focus on Moreton and Amano (1999).
	df = df[df.experimenter == 'MoretonAmano1999']
	xcol = r"C$_1$"
	ycol = r"C$_2$"
	# zcol = "log posterior predictive probability ratio of [a] to [a:]"
	zcol = "Duration threshold between [a] and [a:] (ms)"
	df[xcol] = df.prefix.map(lambda ipa_csv: ipa_csv.replace(',','').split('o')[0])
	df[ycol] = df.prefix.map(lambda ipa_csv: ipa_csv.replace(',','').split('o')[1])
	# df[zcol] = - df.log_post_pred_prob_ratio_target_over_control
	df[zcol] = df.duration_boundary

	ratio_min_dfmaxmin = df.duration_boundary.min() / (df.duration_boundary.max() - df.duration_boundary.min())
	bottom = df[zcol].min() - ratio_min_dfmaxmin * (df[zcol].max() - df[zcol].min())
	# bottom = 0

	get_3dbar(df, xcol, ycol, zcol, bottom = bottom)

	# annotation = (df.prefix.str.replace(',','') + '_').tolist()
	# experiment = 'Duration threshold between [a] and [a:] (msec)'
	# df[experiment] = df.duration_boundary
	# plot_prediction_against_data(df, zcol, experiment, annotation = annotation)
import pandas as pd
import matplotlib.pyplot as plt
import sys, os.path


def plot_results(df, x_label, y_label, group_label, annotation):
	fig, ax = plt.subplots(1, 1)
	cmap = plt.get_cmap('tab10')

	for i, (gp, sub_df) in enumerate(df.groupby(group_label)):
		print i
		sub_df.plot.scatter(x_label, y_label, ax = ax, label = gp, color=cmap(i))

	x_max = df[x_label].max()
	y_max = df[y_label].max()

	if not annotation is None:
		for x, y, a in zip(df[x_label].tolist(), df[y_label].tolist(), annotation):
			x_label_pos = x
			y_label_pos = y
			# if xmax < x + x_max:
			# 	x_label_pos -= x_max/2
			# if ymax < y + y_max:
			# 	y_label_pos -= y_max/2
			ax.annotate(
				a,
				xy = (x, y),
				xytext = (x_label_pos,y_label_pos)
			)
	plt.show()


if __name__ == '__main__':
	df = pd.read_csv(sys.argv[1], encoding='utf-8')

	x_label = 'log p(BASE+sity) - log p(BASE+ness)'
	y_label = 'log p(BASE+kity) - log p(BASE+ness)'

	df[x_label] = df.sity_log_prob - df.ness_log_prob
	df[y_label] = df.kity_log_prob - df.ness_log_prob
	df['type'] = pd.Categorical(df.type, ['Latinate', 'Semi-Latinate', 'Non-Latinate-k', 'Non-Latinate-s'])

	plot_results(df, x_label, y_label, 'type', df.orthography.tolist())
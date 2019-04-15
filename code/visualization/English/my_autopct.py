# coding: utf-8


def my_autopct(pct):
	if pct > 2:
		return "%.2f%%" % pct
	else:
		return ""


def my_autopct_full(pct):
	return "%.2f%%" % pct
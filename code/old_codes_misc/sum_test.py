# coding: utf-8

import numpy as np
import datetime


def sum_pairplus(a1, a2):
	return np.sum(a1+a2)
	
def plus_sum(a1, a2):
	return np.sum(a1)+np.sum(a2)
	
if __name__=='__main__':
	size=(100000,10000)
	a1=np.ones(size)
	a2=np.ones(size)
	start=datetime.datetime.now()
	sum_pairplus(a1, a2)
	print 'sum_pairplus', str(datetime.datetime.now()-start)
	
	start=datetime.datetime.now()
	plus_sum(a1, a2)
	print 'plus_sum', str(datetime.datetime.now()-start)
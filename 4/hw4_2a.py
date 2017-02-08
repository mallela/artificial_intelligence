######################################################
############## Python 2.7#############################
##              CS534 Assignment 4					##
##            Q.No 2(a)								##
##      author: Praneeta Mallela; pmallela@wpi.edu  ##
######################################################

import numpy as np
import csv
import random
import time 

# sigoind function
def sigmoid(x):
  	return 1 / (1 + np.exp(-x))

# hypothesis function
def hyp_fn(w, x):
	z=np.dot(w, x)
	return sigmoid(z)

# compare the old and new weights for gradient descent
def compareWights(old, curr):
	difference = curr - old
	return np.sqrt(np.sum(np.square(difference)))

# get data that's need classsiying from .csv file
def getData(d):
	headers = ([h[0] for h in d][0]).split(',')
	n = len(headers)
	columns = []
	with open(filename,'r') as f:
		reader = csv.DictReader(f)
		g= [r for r in reader]
		f.close()
		for j in range(len(headers)):
			for i in range(len(g)):
				columns.append([v for k,v in g[i].iteritems() if k==headers[j]])
	f.close()
	columns= np.array(columns).reshape(len(headers),len(g))
	conevrt_to_num_one = ['yes', 'f'] 
	for i in range(len(columns)):
		if columns[i][0].isalpha():
			columns[i][:] = [1 if x in conevrt_to_num_one else 0 for x in columns[i][:]]
		else:
			pass
	return columns.astype(np.float)

# this fn performs gradient descent on weights used to classify data
def	gradientDescent(m, n, w, true_y, train_data, lambda_value, learning_rate,epsilon_tolerance):
	flag_converged=0
	counter = 0
	w_old = np.copy(w)
	print "counting no. of iterations",
	while True:
		counter+=1
		hf = hyp_fn(w,train_data)
		temp = np.dot(hf - true_y,train_data[1:,:].T)
		w[0,0] = w[0,0]-(learning_rate/m)*np.sum(hf - true_y)#+(lambda_value/m)*w[0,0]
		for itr in range(n-1):
			w[:,1:] = w[:,1:]-(learning_rate/m)*np.sum(temp)+(lambda_value/m)*sum(w[:,1:])
		if compareWights(w_old, w)<epsilon_tolerance:
			print "\nWeights have converged..!"
			flag_converged = 1
			break
		else:
			w_old = np.copy(w)
			pass
		print '.',
	print "(converged in", counter-1, "iterations)"
	return w

# the initial weights are genertated for n sample data
def generateRandomInitialWeights(n):
	weights =[]
	for i in range(n):
		# weights.append(random.uniform(0,1))
		weights.append(1.0)
	weights = np.reshape(np.array(weights), (1,n))
	return weights

# feature vectors are normalized
def normalizeData(d,n):
	for itr in range(n):
		mean= np.mean(d[itr])
		std = np.std(d[itr]) 
		d[itr] = (d[itr]-mean)/std
	return d

# this fn calls the gradient descent fn to calculate final weights
def findWeights(train_data,m, n, lambda_value, epsilon_tolerance, learning_rate):
	w_x = generateRandomInitialWeights(n)
	true_y = np.reshape(train_data[-1], (1,m))
	w_x = gradientDescent(m,n,w_x, true_y, train_data, lambda_value, learning_rate, epsilon_tolerance)
	return w_x

if __name__=='__main__':
	filename = './data_of_students.csv'
	data = csv.reader(open(filename, 'r'), delimiter = ';')
	cols = getData(data)
	n,m = np.shape(cols)

	cols = normalizeData(cols, n-1) # uncomment this line for standardization
	train_x = np.copy(cols[:n-1,:m])
	alpha = 0.2 # learning rate
	lambda_value = -0.1 # for regularization
	epsilon_tolerance = .5 # convergance criteria
	print "training..."
	t0 = time.time()
	wts = findWeights(train_x, m, n-1, lambda_value, epsilon_tolerance, alpha)
	print "Weights are:\n", wts
	print "Time taken (s):", time.time()-t0

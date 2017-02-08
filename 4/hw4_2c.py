######################################################
############## Python 2.7#############################
##              CS534 Assignment 4					##
##            Q.No 2(c)								##
##      author: Praneeta Mallela; pmallela@wpi.edu  ##
######################################################
import numpy as np
import csv, copy, itertools, math, random
import weka.core.jvm as jvm
import weka.core.converters as converters
from collections import Counter
import matplotlib.pyplot as plt

#Weka is used to retieve data from the .arff file into a .csv file
def convertArff2Csv(infile, outfile):
	jvm.start()
	data = converters.load_any_file(infile)
	converters.save_any_file(data, outfile)
	jvm.stop()

# this fn identifies the missing data positions by finding the data type
def getDataType(data):
	for itr in range(len(data)):
		val_u = data[itr]
		if val_u =='?':
			pass
		else:
		 	return val_u.isalpha()

# filling in missing data - mode for stings and mean for numbers
def bestGuessData(data):
	if getDataType(data): # if it's a string, find mode
		data_count_dict = Counter(data)
		return data_count_dict.most_common(1)[0][0]
	else: # if numbers, find average
		data_wo_missing = [d for d in data if d!='?']
		data_wo_missing = [float(d) for d in data_wo_missing]
		return np.mean(data_wo_missing)

# takes the best guess and replaces the missing data
def replaceMissingData(filename):
	dat = csv.reader(open(filename, 'r'), delimiter = ';')

	headers = ([h[0] for h in dat][0]).split(',')
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
	bg = []
	for i in range(len(headers)):
		bg.append(bestGuessData(columns[i][:]))
		dummy_list= columns[i][:]
		columns[i][:] = [str(bg[i]) if x == '?' else x for x in columns[i][:]]
	dict_str_x = ['normal', 'yes', 'present', 'good','ckd']
	for i in range(len(columns)):
		if columns[i][0].isalpha():
			columns[i][:] = [1 if x in dict_str_x else 0 for x in columns[i][:]]

		else:
			pass
	columns = columns.astype(np.float)
	return columns

def sigmoid(x):
  	return 1 / (1 + np.exp(-x))

# hypohesis function for logit regression
def hyp_fn(w, x):
	z=np.dot(w, x)
	return sigmoid(z)

# compaing old and new weights for convergence criteria
def compareWights(old, curr):
	difference = (curr - old)
	difference= np.sqrt(np.sum(np.square(difference)))
	return difference

# decrease weights usin gradient descent and return new weights
def	gradientDescent(m, n, w, true_y, train_data, lambda_value, learning_rate,epsilon_tolerance):
	flag_converged=0
	counter = 0
	w_old = np.copy(w)
	print "counting iterations...",
	while True:
		counter+=1
		hf = hyp_fn(w,train_data)
		temp = np.dot(hf - true_y,train_data[1:,:].T)
		w[0,0] = w[0,0]-(learning_rate/m)*np.sum(hf - true_y)#+(lambda_value/m)*w[0,0]
		for itr in range(n-1):

			w[:,1:] = w[:,1:]-(learning_rate/m)*np.sum(temp)+(lambda_value/m)*sum(w[:,1:])
		if compareWights(w_old, w)<epsilon_tolerance:
			print "\nWeights have converged", 
			flag_converged = 1
			break
		else:
			w_old = np.copy(w)
			pass
		print ".",
	print "\n(converged in", counter-1, "iterations)"
	return w

# initial weights aer chosen to be "1.0" fo simplicity
def generateRandomInitialWeights(n):
	weights =[]
	for i in range(n):
		weights.append(1.0)
	weights = np.reshape(np.array(weights), (1,n))
	return weights

# normalization of data so data is comparable
def normalizeData(d,n):
	for itr in range(n):
		mean= np.mean(d[itr])
		std = np.std(d[itr]) 
		d[itr] = (d[itr]-mean)/std
	return d

# finding weights needed to classify data
def findWeights(train_data,m, n, lambda_value, epsilon_tolerance, learning_rate):
	w_x = generateRandomInitialWeights(n)
	true_y = np.reshape(train_data[-1], (1,m))
	w_x = gradientDescent(m,n,w_x, true_y, train_data, lambda_value, learning_rate, epsilon_tolerance)
	return w_x

# computation of f measure, returns f measure 
# compares true y with predicted ydef evaluatePridictedWithTrue(predicted_class, true_class):
def evaluatePridictedWithTrue(predicted_class, true_class):
	tp = np.sum([1.0 if (t==1 and p==1) else 0 for t,p in zip(predicted_class.T, true_class.T)])
	print "No. of true positives:", tp# for t,p in zip(predicted_class.T, true_class.T):
	fp = np.sum([1.0 if (p==1 and t==0) else 0 for t,p in zip(predicted_class.T, true_class.T)])
	print "No. of false positives:", fp
	fn = np.sum([1.0 if (p==0 and t==1) else 0 for t,p in zip(predicted_class.T, true_class.T)])
	print "No. of false negatives:", fn
	if (tp+fp)==0:
		return np.inf
	else:
		pre = tp/(tp+fp)
		rec = tp/(tp+fn)
		return 2*pre*rec/(pre+rec)

# predicted y s assessed using f measure,
# and if the predicted y is > .5 the o/p is
# classified as y = 1, and 0 otherwisedef testing(x_test,y_test,wts_test):
def testing(x_test,y_test,wts_test):
	y_predicted = hyp_fn(wts_test, x_test)
	for i in range(80):
		op  = y_predicted[0,i]
		if op>0.5:
			y_predicted[0,i] = 1
		else:
			y_predicted[0,i]=0
	print "Truth value compared to the predicted class and the groud truth is :"
	f_meas=evaluatePridictedWithTrue(y_predicted,y_test)
	return f_meas

if __name__=='__main__':
	# processing the data from files and relevant conversions
	infile  ="./realdata1/chronic_kidney_disease_full.arff"
	outfile = "./realdata1/chronic_kidney_disease_full.csv"
	convertArff2Csv(infile, outfile)
	data_column_wise = replaceMissingData(outfile)	# outfile must be csv
	n, no_samples = np.shape(data_column_wise)
	m1 = int(.1*no_samples)
	m = int(.8*no_samples)
	data_column_wise = normalizeData(data_column_wise, n-1) # uncomment this line for standardization
	train_x = np.copy(data_column_wise[:n-1,m1:m+m1]) # data for training - 80%
	train_y = np.copy(data_column_wise[n-1:,m1:m+m1])
	lambda_range = np.arange(-2.0,4.2,.2)
	f_m = []
	for lambda_value in lambda_range:
		print "lambda is =", lambda_value
		if lambda_value < -0.4 and lambda_value <0:
			alpha = .002 # .01works for (1.0 or .9 ep)
			epsilon_tolerance = .19 #1.6for standardization/normalization
		else: 
			alpha = .002
			epsilon_tolerance = 21
		print "training..." 
		wts = findWeights(train_x, m, n-1, lambda_value, epsilon_tolerance, alpha)
		print "Weights after training are :\n", wts
		################# Testing ######################
		print "testing..."
		test_x = np.copy(data_column_wise[:n-1, :m1]) # data for testing - 20%
		test_x=np.concatenate((test_x,np.copy(data_column_wise[:n-1, m1+m:])), axis=1)
		y = np.copy(data_column_wise[n-1:, :m1]) # true class
		y=np.concatenate((y, np.copy(data_column_wise[n-1:, m1+m:])), axis=1)
		temp_f_m = testing(test_x, y, wts)
		f_m.append(temp_f_m)
		print "f-measure:", temp_f_m
		print "____________________________________________________________________________________"

	# plotting f-measure vs lambda
	plt.figure("f-measure vs lambda")
	plt.title("f-measure vs lambda; alpha = .01")
	plt.plot(lambda_range, f_m, 'ro')
	plt.xlabel("lambda = -2.0 to 4.0 in steps of 0.2")
	plt.ylabel("f-measure")
	plt.ylim((0,1))
	plt.show()
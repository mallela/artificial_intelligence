######################################################
############## Python 2.7#############################
##              CS534 Assignment 4					##
##            Q.No 4								##
##      author: Praneeta Mallela; pmallela@wpi.edu  ##
######################################################
import numpy as np
import csv, copy, itertools, math, random
import weka.core.jvm as jvm
import weka.core.converters as converters
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

# normalization of data so data is comparable
def normalizeData(d,n):
	for itr in range(n):
		mean= np.mean(d[itr])
		std = np.std(d[itr]) 
		d[itr] = (d[itr]-mean)/std
	return d

# computation of f measure, returns f measure 
# compares true y with predicted y
def evaluatePridictedWithTrue(predicted_class, true_class):
	tp = np.sum([1.0 if (t==1 and p==1) else 0 for t,p in zip(predicted_class.T, true_class.T)])
	print "No. of true positives:", tp
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
	
	test_x = np.copy(data_column_wise[:n-1, :m1]) # data for testing - 20%
	test_x=np.concatenate((test_x,np.copy(data_column_wise[:n-1, m1+m:])), axis=1)
	test_y = np.copy(data_column_wise[n-1:, :m1]) # true class
	test_y=np.concatenate((test_y, np.copy(data_column_wise[n-1:, m1+m:])), axis=1)
	
	# SVM train and pridict
	print "SVM with linear kernel"
	clf_svm = SVC(C=1.0, kernel = 'linear',degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
	clf_svm.fit(train_x.T, np.ravel(train_y.T), sample_weight = None)
	predicted_y = clf_svm.predict(test_x.T)
	f_measure_svm = evaluatePridictedWithTrue(predicted_y,test_y)
	print "f-measure:", f_measure_svm
	print 30*"_"
	# 
	print "SVM with RBF kernel"
	clf_svm2 = SVC(C=1.0, kernel = 'rbf',degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
	clf_svm2.fit(train_x.T, np.ravel(train_y.T), sample_weight = None)
	predicted_y2 = clf_svm2.predict(test_x.T)
	f_measure_svm2 = evaluatePridictedWithTrue(predicted_y2,test_y)
	print "f-measure:", f_measure_svm2
	print 30*"_"

	print "Random forest"
	clf_rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
	clf_rf.fit(train_x.T, np.ravel(train_y.T), sample_weight = None)
	predicted_y3 = clf_rf.predict(test_x.T)
	f_measure_svm3 = evaluatePridictedWithTrue(predicted_y3,test_y)
	print "f-measure:", f_measure_svm3
	print 30*"_"

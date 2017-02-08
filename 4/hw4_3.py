######################################################
############## Python 2.7#############################
##              CS534 Assignment 4					##
##            Q.No 3								##
##      author: Praneeta Mallela; pmallela@wpi.edu  ##
######################################################

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import fowlkes_mallows_score
from time import time
import numpy as np
from sklearn.preprocessing import scale
import random 

# generate k random centers
# for ease, 10 random numbers are hosen from the digits datasert (sklearn)
# these centers are then scaled to keep them different from the original data
def randomCenters(k, all_d):
	centers=[]
	for i in range(k):
		centers.append((all_d[random.randrange(0,1796)])*.2)
	return centers

# classifies the data in to k clusters 
# returns classified data and the new cluster centers
def k_means(X, data,k,no_of_iterations):
	k_centers=randomCenters(k, np.copy(X)) # generate k random centers
	for n in range(no_of_iterations):
		for key, value in data.items():
			data[key] = getCostFunction(key,k_centers, X)
		l = len(k_centers)
		for i in range(l):
			same_cluster_data = [k for k,v in data.iteritems() if v==i]
			if same_cluster_data:
				sum_same_clusters = np.sum(X[j] for j in same_cluster_data)
				k_centers[i] = np.copy(sum_same_clusters/len(same_cluster_data))
	return k_centers, data

# returns the cluster that is closest to the given data
def getCostFunction(key_data, k_centers, X):
	dist_list=[]
	for i in range(len(k_centers)):
		dist = np.sum((X[key_data]- k_centers[i])**2)
		dist_list.append(dist)
	return dist_list.index(min(dist_list))

# def missingClusters(cm, no_clusters):
# 	for i in range(no_clusters):
# 		if cm[i,i]==0:
# 			cm[i,i]=-1
# 	return cm

# PROTOCOL1 : counting the max occuring digits in each cluster and returning them as a list
def getClusterRepresentatives(m, k_number):
	reps = []
	for itr in range(k_number):
		r = m[itr,:]
		c = m[:,itr]
		if max(r)>max(c):
			reps.append(list(r).index(max(r)))
		else:
			reps.append(list(c).index(max(c)))
	return reps

if __name__=='__main__':
	digits = load_digits(n_class=10)
	X = digits.data
	y = digits.target
	n_samples, n_features = X.shape
	np.random.seed(0)
	k=10
	labels_y = list(set(y))

	print 50*"_"
	print "KMeans clustering (implementation of algo from question 1a)"
	no_of_iterations = 10
	dat = {i:0 for i in range(n_samples)} 
	t0 = time()
	k_centers, dat = k_means(X, dat, k,no_of_iterations) 
	y_pred1 = [value for key,value in dat.iteritems()]
	c_m1 = confusion_matrix(y,y_pred1, labels_y)	
	print "PROTOCOL1: The cluster predictions for 10 clusters, i.e k = 10 are:\n",getClusterRepresentatives(c_m1, k)
	print "PROTOCOL2: Confusion Matrix: \n",c_m1
	print "PROTOCOL3: Fowlkes-Mallows score:", fowlkes_mallows_score(y, y_pred1)
	print "Time taken: %.2fs" % (time() - t0)

	print 50*"_"
	print "KMeans clustering (using sklearn)"
	clustering1 = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
	t01 = time()
	y_pred11 = clustering1.fit_predict(X)
	c_m11 = confusion_matrix(y,y_pred11)	
	print "PROTOCOL1: The cluster predictions for 10 clusters, i.e k = 10 are:\n",getClusterRepresentatives(c_m11, k)
	print "PROTOCOL2: Confusion Matrix: \n",c_m11
	print "PROTOCOL3: Fowlkes-Mallows score:", fowlkes_mallows_score(y, y_pred11)
	print "Time taken: %.2fs" % (time() - t01)

	print 50*"_"
	print "Agglomerative Clustering with Ward linkage"
	clustering2 = AgglomerativeClustering(linkage='ward', n_clusters=k)
	t1 = time()
	y_pred2 = clustering2.fit_predict(X)
	c_m2 = confusion_matrix(y, y_pred2)
	print "PROTOCOL1: The cluster predictions for 10 clusters, i.e k = 10 are:\n",getClusterRepresentatives(c_m2, k)
	print "PROTOCOL2: Confusion Matrix: \n",c_m2
	print "PROTOCOL3: Fowlkes-Mallows score:", fowlkes_mallows_score(y, y_pred2)
	print "Time taken: %.2fs" % (time() - t1)
	print 50*"_"

	print "Affinity Propagation"
	clustering3 = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=30, copy=True, preference=None, affinity='euclidean', verbose=False)
	t2 = time()
	y_pred3 = clustering3.fit_predict(X)
	c_m3=confusion_matrix(y, y_pred3, labels = labels_y)
	print "PROTOCOL1: The cluster predictions for 10 clusters, i.e k = 10 are:\n",getClusterRepresentatives(c_m3, k)
	print "PROTOCOL2: Confusion Matrix: \n",c_m3
	print "PROTOCOL3: Fowlkes-Mallows score:", fowlkes_mallows_score(y, y_pred3)
	print "Time taken: %.2fs" % (time() - t2)
